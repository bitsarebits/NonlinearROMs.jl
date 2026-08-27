# Helpers and Devices

const CDEV = Lux.cpu_device()
const XDEV = Lux.reactant_device(;force=true)

struct ZscoreStats{A<:AbstractVector,B<:AbstractVector}
  μ::A
  σ::B
end

function ZscoreStats(data::AbstractMatrix;normalise=false)
  if normalise
    stats = ZscoreStats(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  μ = dropdims(mean(data,dims=2),dims=2)
  σ = dropdims(std(data,dims=2),dims=2)
  # Avoid dividing by zero if a feature is constant
  for i in eachindex(σ)
    iszero(σ[i]) && (σ[i] = one(eltype(σ)))
  end
  return ZscoreStats(μ,σ)
end

struct NeuralStats{T<:Real,A<:ZscoreStats,B<:ZscoreStats}
  dmax::T
  pscore::A 
  xscore::B
end

function NeuralStats(data,params,coords;normalise=false)
  dmax = maximum(abs,data)
  normalise && (data ./= dmax)
  input = ZscoreStats(params;normalise)
  output = ZscoreStats(coords;normalise)
  NeuralStats(dmax,input,output)
end

"""
    (model::Lux.AbstractLuxLayer)(inputs,ps,st,stats::NeuralStats)

Applies `model` and denormalises its output using `stats.dmax`, so callers don't need to
separately track and re-apply the target's normalisation scale after inference.
"""
function (model::Lux.AbstractLuxLayer)(inputs,ps,st,stats::NeuralStats)
  pred,st = model(inputs,ps,st)
  pred .*= stats.dmax
  return pred,st
end

# Training loop

"""
    train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger::TrainingLog)

Generic Lux/Reactant/Enzyme training loop shared by DeepONet and NOMAD. `to_device_batch`
maps one raw batch yielded by `dataloader` to the `(inputs,target)` pair (already moved to
`XDEV`) expected by `Lux.Training.single_train_step!`; this is the only piece that differs
between the two architectures (DeepONet pairs each batch with a fixed set of trunk query
points, NOMAD's coordinates are already part of the per-row batch).
"""
function train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger::TrainingLog)
  init!(logger)

  Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH) do
    for epoch in 1:logger.max_epochs
      local current_loss = 0.0f0

      for raw_batch in dataloader
        batch_dev = to_device_batch(raw_batch)

        _,loss_val,_,train_state = Lux.Training.single_train_step!(
          Lux.AutoEnzyme(),
          Lux.MSELoss(),
          batch_dev,
          train_state;
          return_gradients=Val(false)
        )
        current_loss += Float32(loss_val)
      end
      current_loss /= length(dataloader)

      step_scheduler!(lr_scheduler,train_state.optimizer_state,epoch,current_loss;verbose=logger.verbose)

      update!(logger,epoch,current_loss)
    end
  end

  finalize!(logger)
  return train_state.parameters,train_state.states
end

function train_deeponet!(train_state,dataloader,x_data_dev,strategy)
  lr_scheduler = get_scheduler(strategy)
  logger = get_logger(strategy)
  to_device_batch((f_batch,u_batch)) = ((f_batch |> XDEV,x_data_dev),u_batch |> XDEV)
  train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger)
end

function train_nomad!(train_state,dataloader,strategy)
  lr_scheduler = get_scheduler(strategy)
  logger = get_logger(strategy)
  to_device_batch(((u_batch,y_batch),v_batch)) = ((u_batch |> XDEV,y_batch |> XDEV),v_batch |> XDEV)
  train_model!(train_state,dataloader,lr_scheduler,to_device_batch;logger)
end

# Regression networks (hyper-reduction): MultiLayerPerceptron / AutoEncoder

"""
    struct TrainedModel{C,P,S} <: NeuralNetwork
      chain::C
      ps::P
      st::S
    end

A trained Lux `chain` bundled with its optimised parameters/states, evaluable
as `(a::TrainedModel)(x::AbstractMatrix) -> AbstractMatrix` via the standard
`Arrays.evaluate!`/`return_cache` interface. Returned by [`TrainedNeuralNetwork`](@ref)
for [`MultiLayerPerceptron`](@ref) strategies.
"""
struct TrainedModel{C,P,S} <: NeuralNetwork
  chain::C
  ps::P
  st::S
end

Arrays.return_cache(a::TrainedModel,x::AbstractMatrix) = nothing

function Arrays.evaluate!(cache,a::TrainedModel,x::AbstractMatrix)
  first(a.chain(Float32.(x),a.ps,a.st))
end

"""
    TrainedNeuralNetwork(strategy::NeuralOpStrategy,r::AbstractRealisation,coeff) -> NeuralNetwork

Builds and trains a [`NeuralNetwork`](@ref) from `strategy.model`'s recipe and
`(r,coeff)` data, through the same Lux/Reactant/Enzyme pipeline used for DeepONet/NOMAD.
For a [`MultiLayerPerceptron`](@ref), the input/output dimensions are inferred from
`r`/`coeff` and appended to `strategy.model.hidden_layers`; for an [`AutoEncoder`](@ref),
`r` is ignored and the network is trained to reconstruct `coeff`.
"""
function TrainedNeuralNetwork(strategy::NeuralOpStrategy{<:MultiLayerPerceptron},r::AbstractRealisation,coeff)
  x = Float32.(matrix_of_params(r))
  y = Float32.(_get_data(coeff))
  nin,nout = size(x,1),size(y,1)
  n_samples = size(x,2)

  chain = build_lux_chain((nin,strategy.model.hidden_layers...,nout),strategy.model.activation)

  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader((x,y);batchsize=bs,shuffle=true,partial=false)

  Random.seed!(42)
  ps,st = Lux.setup(Random.default_rng(),chain) |> XDEV
  train_state = Lux.Training.TrainState(chain,ps,st,strategy.optimiser.opt)

  to_device_batch((xb,yb)) = (xb |> XDEV,yb |> XDEV)
  ps_trained,st_trained = train_model!(
    train_state,dataloader,strategy.optimiser.lr_scheduler,to_device_batch;logger=strategy.trainlog
  )

  TrainedModel(chain,ps_trained |> CDEV,Lux.testmode(st_trained) |> CDEV)
end

"""
    struct TrainedAutoEncoder{C,P,S} <: NeuralNetwork
      chain::C
      ps::P
      st::S
    end

A trained [`AutoEncoder`](@ref), `chain = Lux.Chain(encoder,decoder)`.
`(a::TrainedAutoEncoder)(z)` applies the **decoder** (latent → high-dim), matching
`evaluate!`'s contract; use [`encode`](@ref)/[`decode`](@ref) to access either half directly.
"""
struct TrainedAutoEncoder{C,P,S} <: NeuralNetwork
  chain::C
  ps::P
  st::S
end

Arrays.return_cache(a::TrainedAutoEncoder,x::AbstractMatrix) = nothing

function Arrays.evaluate!(cache,a::TrainedAutoEncoder,z::AbstractMatrix)
  decode(a,z)
end

function encode(a::TrainedAutoEncoder,X::AbstractMatrix)
  first(a.chain.layers.layer_1(Float32.(X),a.ps.layer_1,a.st.layer_1))
end

function decode(a::TrainedAutoEncoder,Z::AbstractMatrix)
  first(a.chain.layers.layer_2(Float32.(Z),a.ps.layer_2,a.st.layer_2))
end

function TrainedNeuralNetwork(strategy::NeuralOpStrategy{<:AutoEncoder},::AbstractRealisation,coeff)
  X = Float32.(_get_data(coeff))
  nin = size(X,1)
  n_samples = size(X,2)

  hidden = strategy.model.hidden_layers[1:end-1]
  latent_dim = last(strategy.model.hidden_layers)
  encoder = build_lux_chain((nin,hidden...,latent_dim),strategy.model.activation)
  decoder = build_lux_chain((latent_dim,reverse(hidden)...,nin),strategy.model.activation)
  chain = Lux.Chain(encoder,decoder)

  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader((X,X);batchsize=bs,shuffle=true,partial=false)

  Random.seed!(42)
  ps,st = Lux.setup(Random.default_rng(),chain) |> XDEV
  train_state = Lux.Training.TrainState(chain,ps,st,strategy.optimiser.opt)

  to_device_batch((xb,yb)) = (xb |> XDEV,yb |> XDEV)
  ps_trained,st_trained = train_model!(
    train_state,dataloader,strategy.optimiser.lr_scheduler,to_device_batch;logger=strategy.trainlog
  )

  TrainedAutoEncoder(chain,ps_trained |> CDEV,Lux.testmode(st_trained) |> CDEV)
end

"""
    struct TrainedAutoDecoder{D,P,S,L} <: NeuralNetwork
      decoder::D
      ps::P
      st::S
      latent_codes::L
    end

A trained [`AutoDecoder`](@ref). `latent_codes[:,i]` is the learned latent
representation of the `i`-th training snapshot; `evaluate!` applies the decoder
(`latent_dim × k → n_h × k`). Use [`infer_latent`](@ref) to fit a latent code for
an unseen snapshot.
"""
struct TrainedAutoDecoder{D,P,S,L} <: NeuralNetwork
  decoder::D
  ps::P
  st::S
  latent_codes::L
end

Arrays.return_cache(a::TrainedAutoDecoder,z::AbstractMatrix) = nothing

function Arrays.evaluate!(cache,a::TrainedAutoDecoder,z::AbstractMatrix)
  first(a.decoder(Float32.(z),a.ps,a.st))
end

"""
    struct LatentCodeLayer{A<:AbstractMatrix} <: Lux.AbstractLuxLayer
      init_codes::A
    end

A Lux layer with no real input: it ignores whatever it is called with and returns its
`(latent_dim,n_train)` parameter matrix unchanged, so the per-sample latent codes of an
[`AutoDecoder`](@ref) are optimised as ordinary Lux parameters jointly with the decoder.
"""
struct LatentCodeLayer{A<:AbstractMatrix} <: Lux.AbstractLuxLayer
  init_codes::A
end

Lux.initialparameters(rng::Random.AbstractRNG,l::LatentCodeLayer) = (codes=copy(l.init_codes),)
Lux.initialstates(rng::Random.AbstractRNG,l::LatentCodeLayer) = NamedTuple()

(l::LatentCodeLayer)(x,ps,st) = ps.codes,st

function TrainedNeuralNetwork(strategy::NeuralOpStrategy{<:AutoDecoder},::AbstractRealisation,coeff)
  X = Float32.(_get_data(coeff))
  nin = size(X,1)
  n_train = size(X,2)

  hidden = strategy.model.hidden_layers[1:end-1]
  latent_dim = last(strategy.model.hidden_layers)
  decoder = build_lux_chain((latent_dim,reverse(hidden)...,nin),strategy.model.activation)

  Random.seed!(42)
  Z0 = randn(Float32,latent_dim,n_train) .* 0.01f0
  chain = Lux.Chain(LatentCodeLayer(Z0),decoder)

  # Joint decoder + latent-code optimisation is inherently full-batch: every
  # column of the latent-code parameter must be updated on every step.
  dataloader = MLUtils.DataLoader((X,X);batchsize=n_train,shuffle=false,partial=false)

  ps,st = Lux.setup(Random.default_rng(),chain) |> XDEV
  train_state = Lux.Training.TrainState(chain,ps,st,strategy.optimiser.opt)

  to_device_batch((xb,yb)) = (xb |> XDEV,yb |> XDEV)
  ps_trained,st_trained = train_model!(
    train_state,dataloader,strategy.optimiser.lr_scheduler,to_device_batch;logger=strategy.trainlog
  )

  ps_trained_cdev = ps_trained |> CDEV
  TrainedAutoDecoder(
    decoder,ps_trained_cdev.layer_2,Lux.testmode(st_trained).layer_2 |> CDEV,ps_trained_cdev.layer_1.codes
  )
end

"""
    infer_latent(a::TrainedAutoDecoder,x_target::AbstractVector,strategy::NeuralOpStrategy) -> AbstractVector

Fit a latent code `z` for an unseen snapshot `x_target` by minimising the mean
squared reconstruction error with the decoder weights fixed, using `strategy.optimiser.opt`
and `strategy.epochs`.
"""
function infer_latent(a::TrainedAutoDecoder,x_target::AbstractVector,strategy::NeuralOpStrategy)
  latent_dim = size(a.latent_codes,1)
  T = Float32
  z = randn(T,latent_dim) .* T(0.01)
  X_t = reshape(T.(x_target),:,1)
  opt_state = Optimisers.setup(strategy.optimiser.opt,z)
  for _ in 1:strategy.epochs
    grad = ForwardDiff.gradient(z) do z_
      X̂ = first(a.decoder(reshape(z_,:,1),a.ps,a.st))
      sum(abs2,X̂ .- X_t)/length(X_t)
    end
    opt_state,z = Optimisers.update!(opt_state,z,grad)
  end
  z
end

# VariationalAutoEncoder: reparameterisation trick + KL loss. Trained eagerly on
# CPU (no Reactant/XLA tracing), since the reparameterisation step samples fresh
# `randn` values on every forward pass -- under XLA tracing those would be baked
# in as a constant at compile time instead of resampled at each call.

struct VAELayer{E,D} <: Lux.AbstractLuxContainerLayer{(:encoder,:decoder)}
  encoder::E
  decoder::D
  latent_dim::Int
end

function (m::VAELayer)(x,ps,st)
  enc_out,st_enc = m.encoder(x,ps.encoder,st.encoder)
  μ = enc_out[1:m.latent_dim,:]
  log_var = enc_out[m.latent_dim+1:end,:]
  ε = randn(eltype(μ),size(μ))
  z = μ .+ ε .* exp.(log_var ./ 2)
  x̂,st_dec = m.decoder(z,ps.decoder,st.decoder)
  out = vcat(x̂,μ,log_var)
  return out,(encoder=st_enc,decoder=st_dec)
end

function train_vae!(train_state,dataloader,lr_scheduler,loss_fn;logger::TrainingLog)
  init!(logger)

  for epoch in 1:logger.max_epochs
    local current_loss = 0.0f0

    for x_batch in dataloader
      _,loss_val,_,train_state = Lux.Training.single_train_step!(
        Lux.AutoEnzyme(),
        loss_fn,
        x_batch,
        train_state;
        return_gradients=Val(false)
      )
      current_loss += Float32(loss_val)
    end
    current_loss /= length(dataloader)

    step_scheduler!(lr_scheduler,train_state.optimizer_state,epoch,current_loss;verbose=logger.verbose)

    update!(logger,epoch,current_loss)
  end

  finalize!(logger)
  return train_state.parameters,train_state.states
end

"""
    struct TrainedVAE{E,D,PE,SE,PD,SD} <: NeuralNetwork
      encoder::E
      decoder::D
      ps_enc::PE
      st_enc::SE
      ps_dec::PD
      st_dec::SD
      latent_dim::Int
    end

A trained [`VariationalAutoEncoder`](@ref). `evaluate!(cache,a,z)` applies the
**decoder** (latent → high-dim); use [`encode`](@ref) for the encoder direction,
which returns `(μ,log_var,z)` with a freshly sampled `z`.
"""
struct TrainedVAE{E,D,PE,SE,PD,SD} <: NeuralNetwork
  encoder::E
  decoder::D
  ps_enc::PE
  st_enc::SE
  ps_dec::PD
  st_dec::SD
  latent_dim::Int
end

Arrays.return_cache(a::TrainedVAE,z::AbstractMatrix) = nothing

function Arrays.evaluate!(cache,a::TrainedVAE,z::AbstractMatrix)
  decode(a,z)
end

function encode(a::TrainedVAE,X::AbstractMatrix)
  enc_out = first(a.encoder(Float32.(X),a.ps_enc,a.st_enc))
  μ = enc_out[1:a.latent_dim,:]
  log_var = enc_out[a.latent_dim+1:end,:]
  ε = randn(eltype(μ),size(μ))
  z = μ .+ ε .* exp.(log_var ./ 2)
  (μ,log_var,z)
end

function decode(a::TrainedVAE,Z::AbstractMatrix)
  first(a.decoder(Float32.(Z),a.ps_dec,a.st_dec))
end

function TrainedNeuralNetwork(strategy::NeuralOpStrategy{<:VariationalAutoEncoder},::AbstractRealisation,coeff)
  X = Float32.(_get_data(coeff))
  nin = size(X,1)
  n_samples = size(X,2)

  hidden = strategy.model.hidden_layers[1:end-1]
  latent_dim = last(strategy.model.hidden_layers)
  encoder = build_lux_chain((nin,hidden...,2*latent_dim),strategy.model.activation)
  decoder = build_lux_chain((latent_dim,reverse(hidden)...,nin),strategy.model.activation)
  vae = VAELayer(encoder,decoder,latent_dim)

  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader(X;batchsize=bs,shuffle=true,partial=false)

  Random.seed!(42)
  ps,st = Lux.setup(Random.default_rng(),vae)
  train_state = Lux.Training.TrainState(vae,ps,st,strategy.optimiser.opt)

  β = strategy.model.β
  function vae_loss(model::VAELayer,ps,st,x)
    n_h = size(x,1)
    out,st = model(x,ps,st)
    x̂ = view(out,1:n_h,:)
    μ = view(out,n_h+1:n_h+model.latent_dim,:)
    log_var = view(out,n_h+model.latent_dim+1:size(out,1),:)
    recon = sum(abs2,x̂ .- x)/length(x)
    kl = -sum(1 .+ log_var .- μ.^2 .- exp.(log_var))/(2*size(x,2))
    return recon + β*kl,st,(;)
  end

  ps_trained,st_trained = train_vae!(
    train_state,dataloader,strategy.optimiser.lr_scheduler,vae_loss;logger=strategy.trainlog
  )

  TrainedVAE(
    encoder,decoder,
    ps_trained.encoder,Lux.testmode(st_trained.encoder),
    ps_trained.decoder,Lux.testmode(st_trained.decoder),
    latent_dim
  )
end

# Generic Dispatch (Steady)

function train_neural_operator(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  strategy = get_strategy(red)

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(get_sampler(strategy),sx)
  data,params,coords = get_formatted_data(Float32,target)

  # Normalisation
  stats = NeuralStats(data,params,coords;normalise=true)

  # Building the DeepONet
  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(strategy)
  opt = get_optimiser(strategy)
  coords_dev = coords |> XDEV
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # Dataloader and setup
  bs = resolve_batch_size(strategy,num_params(s))
  dataloader = MLUtils.DataLoader(
    (params,data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Executing the pipeline
  ps_trained,st_trained = train_deeponet!(train_state,dataloader,coords_dev,strategy)
  st_test = Lux.testmode(st_trained) |> CDEV

  return model,ps_trained |> CDEV,st_test,stats
end

function train_neural_operator(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool=false
  )

  strategy = get_strategy(red)

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(get_sampler(strategy),sx)
  data,params,coords = get_formatted_data(Float32,target)

  # Normalisation
  if update_stats
    stats = NeuralStats(data,params,coords;normalise=true)
  else
    stats = pretrained_op.norm_stats
    normalise!((data,params,coords),stats)
  end

  # Pretrained model
  model = pretrained_op.model
  opt = get_optimiser(strategy)
  coords_dev = coords |> XDEV
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # Dataloader and setup
  bs = resolve_batch_size(strategy,num_params(s))
  dataloader = MLUtils.DataLoader(
    (params,data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Executing the pipeline
  ps_trained,st_trained = train_deeponet!(train_state,dataloader,coords_dev,strategy)
  st_test = Lux.testmode(st_trained) |> CDEV

  return model,ps_trained |> CDEV,st_test,stats
end

function train_neural_operator(
  red::NOMADReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  strategy = get_strategy(red)

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(get_sampler(strategy),sx)
  data,params,coords = get_formatted_data(Float32,target)
  dout,pin,xin = _flatten(data,params,coords) # Flattening for NOMAD
  N_tot = size(dout,2)

  # Normalisation
  stats = NeuralStats(dout,pin,xin;normalise=true)

  # Building the NOMAD model
  rng = Random.default_rng()
  Random.seed!(rng,42)

  model = build_model(strategy)
  opt = get_optimiser(strategy)
  ps,st = Lux.setup(rng,model) |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # DataLoader and Lux setup
  bs = resolve_batch_size(strategy,N_tot)
  dataloader = MLUtils.DataLoader(
    ((pin,xin),dout);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Running the pipeline
  ps_trained,st_trained = train_nomad!(train_state,dataloader,strategy)
  st_test = Lux.testmode(st_trained) |> CDEV

  return model,ps_trained |> CDEV,st_test,stats
end

function train_neural_operator(
  red::NOMADReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
  )

  strategy = get_strategy(red)

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(get_sampler(strategy),sx)
  data,params,coords = get_formatted_data(Float32,target)
  dout,pin,xin = _flatten(data,params,coords) # Flattening for NOMAD
  N_tot = size(dout,2)

  # Normalisation
  if update_stats
    stats = NeuralStats(dout,pin,xin;normalise=true)
  else
    stats = pretrained_op.norm_stats
    normalise!((dout,pin,xin),stats)
  end

  # Pretrained model
  model = pretrained_op.model
  opt = get_optimiser(strategy)
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV
  train_state = Lux.Training.TrainState(model,ps,st,opt)

  # DataLoader and Lux setup
  bs = resolve_batch_size(strategy,N_tot)
  dataloader = MLUtils.DataLoader(
    ((pin,xin),dout);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  # Running the pipeline
  ps_trained,st_trained = train_nomad!(train_state,dataloader,strategy)
  st_test = Lux.testmode(st_trained) |> CDEV

  return model,ps_trained |> CDEV,st_test,stats
end

# utils

function resolve_batch_size(batch_config::Int,total_samples::Int)
  return batch_config <= 0 ? total_samples : min(batch_config,total_samples)
end

function resolve_batch_size(strategy::NeuralOpStrategy,total_samples::Int)
  resolve_batch_size(strategy.batch_size,total_samples)
end

normalise!(args...) = @abstractmethod

function normalise!(data::AbstractVector,stats::ZscoreStats)
  data .-= stats.μ
  data ./= stats.σ
  data
end

function normalise!(data::AbstractMatrix,stats::ZscoreStats)
  for v in eachcol(data)
    normalise!(v,stats)
  end
  data
end

function normalise!(inout::NTuple{2,AbstractArray},stats::NeuralStats)
  a,b = inout
  normalise!(a,stats.pscore)
  normalise!(b,stats.xscore)
end

function normalise!(inout::NTuple{3,AbstractArray},stats::NeuralStats)
  a,b,c = inout
  a ./= stats.dmax
  normalise!(b,stats.xscore)
  normalise!(c,stats.dscore)
end

function _flatten(
  data::AbstractArray{T},
  params::AbstractArray{T},
  coords::AbstractArray{T}
  ) where T 

  ntot = size(coords,2)*size(params,2)
  pin = zeros(T,size(params,1),ntot)
  xin = zeros(T,size(coords,1),ntot)
  dout = zeros(T,1,ntot)

  col_idx = 1
  @views for i in axes(params,2)
    p = params[:,i]
    for j in axes(coords,2)
      pin[:,col_idx] = p
      xin[:,col_idx] = coords[:,j]
      dout[1,col_idx] = data[j,i]
      col_idx += 1
    end
  end

  return dout,pin,xin
end

function _flatten(
  params::AbstractArray{T},
  coords::AbstractArray{T}
  ) where T

  ntot = size(coords,2)*size(params,2)
  pin = zeros(T,size(params,1),ntot)
  xin = zeros(T,size(coords,1),ntot)

  col_idx = 1
  @views for i in axes(params,2)
    p = params[:,i]
    for j in axes(coords,2)
      pin[:,col_idx] = p
      xin[:,col_idx] = coords[:,j]
      col_idx += 1
    end
  end

  return pin,xin
end