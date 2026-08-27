# Helpers and Devices

const CDEV = Lux.cpu_device()
const XDEV = Lux.reactant_device(;force=true)

# Training calls

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

"""
    TrainedNeuralNetwork(strategy::NeuralStrategy,r::AbstractRealisation,coeff) -> NeuralNetwork

Builds and trains a [`NeuralNetwork`](@ref) from `strategy.model`'s recipe and
`(r,coeff)` data, through the same Lux/Reactant/Enzyme pipeline used for DeepONet/NOMAD.
For a [`MultiLayerPerceptron`](@ref), the input/output dimensions are inferred from
`r`/`coeff` and appended to `strategy.model.hidden_layers`; for an [`AutoEncoder`](@ref),
`r` is ignored and the network is trained to reconstruct `coeff`.
"""
function TrainedNeuralNetwork(strategy::NeuralStrategy{<:MultiLayerPerceptron},r::AbstractRealisation,coeff)
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
  train_model!(
    train_state,dataloader,strategy.optimiser.lr_scheduler,to_device_batch;logger=strategy.trainlog
  )
end

function TrainedNeuralNetwork(strategy::NeuralStrategy{<:AutoEncoder},::AbstractRealisation,coeff)
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
  train_model!(
    train_state,dataloader,strategy.optimiser.lr_scheduler,to_device_batch;logger=strategy.trainlog
  )
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

function TrainedNeuralNetwork(strategy::NeuralStrategy{<:AutoDecoder},::AbstractRealisation,coeff)
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
  train_model!(
    train_state,dataloader,strategy.optimiser.lr_scheduler,to_device_batch;logger=strategy.trainlog
  )
end

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

function TrainedNeuralNetwork(strategy::NeuralStrategy{<:VariationalAutoEncoder},::AbstractRealisation,coeff)
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
  stats = NormStats(data,params,coords;normalise=true)

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
  trained = train_deeponet!(train_state,dataloader,coords_dev,strategy)

  return model,trained.parameters,trained.states,stats
end

function train_neural_operator(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  strategy = get_strategy(red)

  # Data extraction
  sx = CoordinateSnapshots(s,get_test(feop))
  target = sample(get_sampler(strategy),sx)
  data,params,coords = get_formatted_data(Float32,target)

  # Normalisation
  if update_stats
    stats = NormStats(data,params,coords;normalise=true)
  else
    stats = pretrained_op.norm_stats
    expected_branch_in = length(stats.pscore.μ)
    expected_trunk_in = length(stats.xscore.μ)
    @assert size(params,1) == expected_branch_in "Branch dimension mismatch: expected $expected_branch_in, got $(size(params,1)). Check the parameter sampler."
    @assert size(coords,1) == expected_trunk_in "Trunk dimension mismatch: expected $expected_trunk_in, got $(size(coords,1))."
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
  trained = train_deeponet!(train_state,dataloader,coords_dev,strategy)

  return model,trained.parameters,trained.states,stats
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
  stats = NormStats(dout,pin,xin;normalise=true)

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
  trained = train_nomad!(train_state,dataloader,strategy)

  return trained,stats
end

function train_neural_operator(
  red::NOMADReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
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
    stats = NormStats(dout,pin,xin;normalise=true)
  else
    stats = pretrained_op.norm_stats
    expected_sensors = length(stats.pscore.μ)
    expected_coords = length(stats.xscore.μ)
    @assert size(pin,1) == expected_sensors "Sensors input dimension mismatch: expected $expected_sensors, got $(size(pin,1))."
    @assert size(xin,1) == expected_coords "Coords input dimension mismatch: expected $expected_coords, got $(size(xin,1))."
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
  trained = train_nomad!(train_state,dataloader,strategy)

  return trained,stats
end

# utils

function resolve_batch_size(batch_config::Int,total_samples::Int)
  return batch_config <= 0 ? total_samples : min(batch_config,total_samples)
end

function resolve_batch_size(strategy::NeuralStrategy,total_samples::Int)
  resolve_batch_size(strategy.batch_size,total_samples)
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