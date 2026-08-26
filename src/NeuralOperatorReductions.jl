struct NeuralOptimiser
  opt::Optimisers.AbstractRule 
  lr_scheduler::LRScheduler
end

function NeuralOptimiser(;
  lr_scheduler::LRScheduler,
  opt=Optimisers.Adam(get_lr(lr_scheduler)),
  weight_decay::Real=0.0
  )

  opt = weight_decay > 0 ? Optimisers.OptimiserChain(opt,Optimisers.WeightDecay(weight_decay)) : opt
  NeuralOptimiser(opt,lr_scheduler)
end

"""
    Base.@kwdef struct NeuralOpStrategy{M,S}
      model::M
      epochs::Int = 20000
      batch_size::Int = 0
      space_step = 1
      param_step = 1
      time_step = nothing
      lr_scheduler::S = CosineAnnealing(epochs)
      verbose::Bool = true
      print_every::Int = 500
    end

The central configuration struct for training Neural Operators. It defines the
neural architecture, the training hyperparameters, and the data subsampling
strategies for the offline phase.

# Fields
- `model`: A [`DeepONet`](@ref) or [`NOMAD`](@ref) architecture. Build one either with
  explicit `branch_layers`/`trunk_layers` (or `approximator_layers`/`decoder_layers`), or
  via the convenience `DeepONet(nbranch_in, ntrunk_in; width, depth, activation)` /
  `NOMAD(nsensors_in, ncoords_in; width, depth, activation)` constructors, which build a
  uniform stack of `depth` hidden layers of `width` neurons from the given input dimensions.
- `epochs::Int`: Total number of training epochs. Default: `20000`.
- `batch_size::Int`: The batch size for training. If set to `0` or a negative value, it defaults to the total number of available samples (full-batch). Default: `0`.
- `sampler::NeuralSampler`: Built from `space_step`/`param_step`/`time_step`, controls how the
  spatial DoFs, the parameters, and (for transient problems) the time steps are subsampled
  from the full-order data before feeding it to the network. Each of `space_step`/`param_step`/
  `time_step` can be an `Integer` (stride), a `Function` (a per-sample transform, e.g.
  `p -> log10.(p)`), an `AbstractVector` of explicit indices, or `nothing`/`identity` (no
  subsampling). Default: `space_step=1`, `param_step=1`, `time_step=nothing`.
- `lr_scheduler`: The learning rate scheduler to use (e.g., `CosineAnnealing(epochs)`, `ReduceLROnPlateau()`). Default: `CosineAnnealing(epochs)`.
- `verbose::Bool`: If `true`, prints compilation times, training progress, and loss metrics. Default: `true`.
- `print_every::Int`: Frequency (in epochs) of the training progress output. Default: `500`.

# Examples

**Basic Usage:**
```julia
using Lux

strategy = NeuralOpStrategy(
  DeepONet(2, 2; width=64, depth=3, activation=Lux.gelu), # 2 params -> Branch; 2D coords -> Trunk
  epochs = 5000,
  batch_size = 32,
  space_step = 2, # Use half of the spatial DoFs for training
  lr_scheduler = CosineAnnealing(5000, lr_max=1e-3, lr_min=1e-6)
  )
```

**Advanced Usage (Multi-Scale Learning):**
```julia
# Log-transform for parameters spanning huge ranges (e.g., 1e-(beta) with beta = 1:0.2:5)
strategy_log = NeuralOpStrategy(
  DeepONet(2, 3; width=64, depth=3), # 2 params -> Branch; 3D coords -> Trunk
  param_step = p -> log10.(p)
  )
```
"""
struct NeuralOpStrategy{A<:NeuralNetwork}
  model::A
  epochs::Int
  batch_size::Int
  sampler::NeuralSampler
  optimiser::NeuralOptimiser
  trainlog::TrainingLog
end

function NeuralOpStrategy(
  model::NeuralNetwork;
  epochs::Int=20000,
  batch_size::Int=0,
  space_step=1,
  param_step=1,
  time_step=nothing,
  lr_scheduler=CosineAnnealing(epochs),
  verbose::Bool=true,
  print_every::Int=500,
  kwargs...
  )

  sampler = NeuralSampler(;space_step,param_step,time_step)
  optimiser = NeuralOptimiser(;lr_scheduler,kwargs...)
  name = string(nameof(typeof(model)))
  trainlog = TrainingLog(name,epochs;verbose,print_every)
  NeuralOpStrategy(
    model,
    epochs,
    batch_size,
    sampler,
    optimiser,
    trainlog
  )
end

struct NeuralOpReduction{A<:NeuralNetwork} <: Reduction{NoReductionStyle,EuclideanNorm}
  strategy::NeuralOpStrategy{A}
end

RBSteady.ReductionStyle(r::NeuralOpReduction) = NoReductionStyle()
RBSteady.NormStyle(r::NeuralOpReduction) = EuclideanNorm()
get_strategy(r::NeuralOpReduction) = r.strategy

"""
    const DeepONetReduction{M<:DeepONet} = NeuralOpReduction{M}

A reduction wrapper for the Deep Operator Network (DeepONet) strategy.
It instructs the ROM solvers to use the DeepONet pipeline during the offline and online phases.

# Constructors
- `DeepONetReduction(s::NeuralOpStrategy)`: Wraps an explicitly defined strategy.
- `DeepONetReduction(; model::DeepONet, kwargs...)`: Automatically builds the strategy, forwarding the training-hyperparameter keyword arguments to [`NeuralOpStrategy`](@ref).

# Examples
```julia
# Using an explicit strategy
strategy = NeuralOpStrategy(DeepONet(2,3;width=64,depth=3), epochs=1000)
reduction = DeepONetReduction(strategy)

# Using kwargs directly
reduction = DeepONetReduction(model=DeepONet(2,3;width=64,depth=3), epochs=1000, batch_size=32)
```
"""
const DeepONetReduction{M<:DeepONet} = NeuralOpReduction{M}

"""
    const NOMADReduction{M<:NOMAD} = NeuralOpReduction{M}

A reduction wrapper for the NOMAD (Non-linear Manifold Decoder) neural operator strategy.
It instructs the ROM solvers to use the NOMAD pipeline during the offline and online phases.

# Constructors
- `NOMADReduction(s::NeuralOpStrategy)`: Wraps an explicitly defined strategy.
- `NOMADReduction(; model::NOMAD, kwargs...)`: Automatically builds the strategy, forwarding the training-hyperparameter keyword arguments to [`NeuralOpStrategy`](@ref).

# Examples
```julia
# Using an explicit strategy
strategy = NeuralOpStrategy(NOMAD(2,3;width=32,depth=2), epochs=1000)
reduction = NOMADReduction(strategy)

# Using kwargs directly
reduction = NOMADReduction(model=NOMAD(2,3;width=32,depth=2), epochs=1000)
```
"""
const NOMADReduction{M<:NOMAD} = NeuralOpReduction{M}

for (f,m) in ((:DeepONetReduction,:DeepONet),(:NOMADReduction,:NOMAD))
  @eval begin
    $f(s::NeuralOpStrategy{<:$m}) = NeuralOpReduction(s)

    function $f(;model::$m,kwargs...)
      strategy = NeuralOpStrategy(model;kwargs...)
      NeuralOpReduction(strategy)
    end
  end
end

"""
    const NeuralOpSolver{A,C<:NeuralOpReduction} = GlobalRBSolver{A,C,Nothing,Nothing}

    NeuralOpSolver(fesolver::GridapType, reduction::NeuralOpReduction)

Initializes the Reduced Basis Solver for Neural Operators.

# Arguments
- `fesolver`: The high-fidelity standard Gridap solver (e.g., `LUSolver()`). In the context of Reduced Order Models, the neural operator acts as a surrogate for this specific full-order solver. This reference defines the underlying high-fidelity model being approximated.
- `reduction::NeuralOpReduction`: The configured neural reduction strategy (e.g., `DeepONetReduction` or `NOMADReduction`).

# Examples

**Minimal Default Initialization:**
```julia
# Default hyperparameters (20000 epochs, full-batch, etc.), 2 params -> Branch, 2D coords -> Trunk
solver = NeuralOpSolver(LUSolver(), DeepONetReduction(model=DeepONet(2,2)))
```
**Custom Initialization:**
```julia
using Lux

strategy = NeuralOpStrategy(
  DeepONet(2, 2; width=128, depth=4, activation=Lux.gelu),
  epochs = 1000
  )
reduction = DeepONetReduction(strategy)
solver = NeuralOpSolver(ThetaMethod(LUSolver(), dt, θ), reduction)
```
"""
const NeuralOpSolver{A,C<:NeuralOpReduction} = GlobalRBSolver{A,C,Nothing,Nothing}

function NeuralOpSolver(fesolver,reduction::NeuralOpReduction)
  RBSolver(fesolver,GlobalContext(),reduction,nothing,nothing)
end

"""
    struct NeuralRBOperator{O,T,Mod,M,S,NStats,U} <: RBOperator{O,T}
      op::ParamOperator{O,T}
      model::Mod
      model_weights::M
      model_states::S
      norm_stats::NStats
      max_u::U
    end

The evaluated Reduced Basis Operator for Neural Operators.
This struct is the direct output of the offline training phase and is passed to the `solve` function during the online phase.

It stores the high-fidelity operator, the trained model, the optimized network weights and states, and the normalization statistics used to scale the data.

# Fields
- `op`: The original high-fidelity parametric operator.
- `model`: The trained Neural Operator architecture.
- `model_weights`: The optimized weights of the network.
- `model_states`: The states of the network (e.g., Batch Normalization running averages, if any).
- `norm_stats`: A NamedTuple containing the z-score statistics (\$\\mu\$ and \$\\sigma\$) computed during training to normalize the inputs.
- `max_u`: The absolute maximum scalar value of the snapshot target data, used for the final denormalization of the network predictions.
"""
struct NeuralRBOperator{O,T,Mod,M,S,NStats,U} <: RBOperator{O,T}
  op::ParamOperator{O,T}
  model::Mod
  model_weights::M
  model_states::S
  norm_stats::NStats
  max_u::U
end

ParamSteady.get_fe_operator(op::NeuralRBOperator) = op.op
