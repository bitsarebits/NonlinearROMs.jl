
# Train model 

"""
    struct TrainedModel{A,B,C} <: NeuralNetwork
      chain::A
      parameters::B
      states::C
    end

A trained Lux `chain` bundled with its optimised parameters/states, evaluable
as `(a::TrainedModel)(x::AbstractMatrix) -> AbstractMatrix` via the standard
`Arrays.evaluate!`/`return_cache` interface. Returned by [`TrainedNeuralNetwork`](@ref)
for [`MultiLayerPerceptron`](@ref) strategies.
"""
struct TrainedModel{A,B,C} <: NeuralNetwork
  chain::A
  parameters::B
  states::C
end

function TrainedModel(train_state::Lux.Training.TrainState)
  parameters = train_state.parameters |> CDEV
  states = Lux.testmode(train_state.states) |> CDEV
  TrainedModel(train_state.model,parameters,states)
end

function Arrays.evaluate!(cache,a::TrainedModel,x::AbstractMatrix)
  first(a.chain(Float32.(x),a.parameters,a.states))
end

"""
    (m::TrainedModel)(inputs) -> AbstractArray
    (m::TrainedModel)(inputs,metadata) -> AbstractArray

Applies `m` to `inputs` (a `(params,coords)`/`(pin,xin)` tuple for DeepONet/NOMAD, or
a plain matrix). `metadata` optionally denormalises the output by `metadata.dmax`; it
is a no-op when `metadata === nothing` (a `NeuralOperator` with no normalisation stats).
"""
function (m::TrainedModel)(inputs)
  first(m.chain(inputs,m.parameters,m.states))
end

(m::TrainedModel)(inputs,metadata::Nothing) = m(inputs)

function (m::TrainedModel)(inputs,metadata::NormStats)
  pred = m(inputs)
  pred .*= metadata.dmax
  return pred
end

const TrainedAutoEncoder = TrainedModel{<:AutoEncoder}

function Arrays.evaluate!(cache,a::TrainedAutoEncoder,z::AbstractMatrix)
  decode(a,z)
end

function encode(a::TrainedAutoEncoder,X::AbstractMatrix)
  first(a.chain.layers.layer_1(Float32.(X),a.parameters.layer_1,a.states.layer_1))
end

function decode(a::TrainedAutoEncoder,Z::AbstractMatrix)
  first(a.chain.layers.layer_2(Float32.(Z),a.parameters.layer_2,a.states.layer_2))
end

const TrainedAutoDecoder = TrainedModel{<:AutoDecoder}

function Arrays.evaluate!(cache,a::TrainedAutoDecoder,z::AbstractMatrix)
  first(a.decoder(Float32.(z),a.parameters,a.states))
end

get_latent_codes(a::TrainedAutoDecoder) = a.parameters.layer_1.codes

"""
    infer_latent(a::TrainedAutoDecoder,x_target::AbstractVector,strategy::NeuralStrategy) -> AbstractVector

Fit a latent code `z` for an unseen snapshot `x_target` by minimising the mean
squared reconstruction error with the decoder weights fixed, using `strategy.optimiser.opt`
and `strategy.epochs`.
"""
function infer_latent(a::TrainedAutoDecoder,x_target::AbstractVector,strategy::NeuralStrategy)
  latent_dim = size(get_latent_codes(a),1)
  T = Float32
  z = randn(T,latent_dim) .* T(0.01)
  X_t = reshape(T.(x_target),:,1)
  opt_state = Optimisers.setup(strategy.optimiser.opt,z)
  for _ in 1:strategy.epochs
    grad = ForwardDiff.gradient(z) do z_
      X̂ = first(a.decoder(reshape(z_,:,1),a.parameters,a.states))
      sum(abs2,X̂ .- X_t)/length(X_t)
    end
    opt_state,z = Optimisers.update!(opt_state,z,grad)
  end
  z
end

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
  return TrainedModel(train_state)
end
