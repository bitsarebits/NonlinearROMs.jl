"""
    Base.@kwdef struct NeuralOpStrategy{M,S}
      model::M
      epochs::Int = 20000
      batch_size::Int = 0
      step_x::Int = 1
      step_t::Int = 1
      branch_sampler::Function = identity
      lr_scheduler::S = CosineAnnealing(epochs)
      verbose::Bool = true
      print_every::Int = 500
    end

The central configuration struct for training Neural Operators. It defines the
neural architecture, the training hyperparameters, and the data subsampling
strategies for the offline phase.

# Fields
- `model`: The neural network architecture builder (e.g., `AutoDeepONet()`, `AutoNOMAD()` for default architectures).
- `epochs::Int`: Total number of training epochs. Default: `20000`.
- `batch_size::Int`: The batch size for training. If set to `0` or a negative value, it defaults to the total number of available samples (full-batch). Default: `0`.
- `step_x::Int`: Spatial subsampling step. Extracts 1 DoF every `step_x` for the Trunk/Coordinate inputs. Useful for reducing memory footprints in dense meshes. Default: `1`.
- `step_t::Int`: Temporal subsampling step (used only for transient problems). Default: `1`.
- `branch_sampler::Function`: A custom function to preprocess or extract specific sensor values/parameters from the raw parametric matrix before feeding them to the Branch Net (or NOMAD sensors).
  It receives a single parameter vector `p` for each sample. This is extremely powerful for:
  1. **Multi-Scale Learning:** Applying transformations (e.g., `p -> log10.(p)`) to handle parameters spanning several orders of magnitude before the automatic Z-score normalization.
  2. **Multi-Sensor/Multi-Function inputs:** Unpacking multiple parameters to sample different continuous functions, concatenating the results into a single 1D vector (see examples).
  Default: `identity`.
- `lr_scheduler`: The learning rate scheduler to use (e.g., `CosineAnnealing(epochs)`, `ReduceLROnPlateau()`). Default: `CosineAnnealing(epochs)`.
- `verbose::Bool`: If `true`, prints compilation times, training progress, and loss metrics. Default: `true`.
- `print_every::Int`: Frequency (in epochs) of the training progress output. Default: `500`.

# Examples

**Basic Usage:**
```julia
using Lux

strategy = NeuralOpStrategy(
  model = AutoDeepONet(width=64, depth=3, activation=Lux.gelu),
  epochs = 5000,
  batch_size = 32,
  step_x = 2, # Use half of the spatial DoFs for training
  lr_scheduler = CosineAnnealing(5000, lr_max=1e-3, lr_min=1e-6)
)
```

**Advanced Usage (Multi-Sensor & Multi-Scale):**
```julia
# Log-transform for parameters spanning huge ranges (e.g., 1e-(beta) with beta = 1:0.2:5)
strategy_log = NeuralOpStrategy(
  model = AutoDeepONet(),
  branch_sampler = p -> log10.(p)
)

# Multi-parameter sampling mapping into concatenated sensor functions
m_sensors = 100

# x_sensors does not need to be a uniform grid. It can be any array of coordinates,
# for example, denser around localized features or sharp gradients.
x_sensors = range(0, 1, length=m_sensors)

f1(x, sigma) = (1 / √(2 * π * sigma)) * exp(-x^2 / (2 * sigma))
f2(x, sigma) = sigma * x^2
f3(x, mu)    = sin(mu * x)

branch_sampler_func = (p) -> begin
    sigma, mu = p[1], p[2]

    # Sample the functions using the current parameters
    sensors_f1 = [f1(x, sigma) for x in x_sensors]
    sensors_f2 = [f2(x, sigma) for x in x_sensors]
    sensors_f3 = [f3(x, mu) for x in x_sensors]

    # IMPORTANT: The Branch Net expects a single flat 1D vector per sample.
    # You must concatenate all sensor arrays using `vcat`.
    return vcat(sensors_f1, sensors_f2, sensors_f3) # Returns a flat 1D vector of length 300
end

strategy_multi = NeuralOpStrategy(
  model = AutoDeepONet(),
  branch_sampler = branch_sampler_func
)
```
"""
Base.@kwdef struct NeuralOpStrategy{M,S}
  model::M
  epochs::Int = 20000
  batch_size::Int = 0
  step_x::Int = 1
  step_t::Int = 1
  branch_sampler::Function = identity
  lr_scheduler::S = CosineAnnealing(epochs)
  verbose::Bool = true
  print_every::Int = 500
end

struct NeuralOpReduction{M,S} <: Reduction{NoReductionStyle,EuclideanNorm}
  strategy::NeuralOpStrategy{M,S}
end

RBSteady.ReductionStyle(r::NeuralOpReduction) = NoReductionStyle()
RBSteady.NormStyle(r::NeuralOpReduction) = EuclideanNorm()
ParamDataStructures.num_params(r::NeuralOpReduction) = 1 # Dummy value
get_strategy(r::NeuralOpReduction) = r.strategy

"""
    const DeepONetReduction{M<:AbstractDeepONet,S} = NeuralOpReduction{M,S}

A reduction wrapper for the Deep Operator Network (DeepONet) strategy.
It instructs the ROM solvers to use the DeepONet pipeline during the offline and online phases.

# Constructors
- `DeepONetReduction(s::NeuralOpStrategy)`: Wraps an explicitly defined strategy.
- `DeepONetReduction(; model = AutoDeepONet(), kwargs...)`: Automatically builds the strategy forwarding the keyword arguments.

# Examples
```julia
# Using an explicit strategy
strategy = NeuralOpStrategy(model=AutoDeepONet(), epochs=1000)
reduction = DeepONetReduction(strategy)

# Using kwargs directly
reduction = DeepONetReduction(model=AutoDeepONet(), epochs=1000, batch_size=32)
```
"""
const DeepONetReduction{M<:AbstractDeepONet,S} = NeuralOpReduction{M,S}

DeepONetReduction(s::NeuralOpStrategy{<:AbstractDeepONet}) = NeuralOpReduction(s)
function DeepONetReduction(;model = AutoDeepONet(),kwargs...)
  NeuralOpReduction(NeuralOpStrategy(;model=model,kwargs...))
end

"""
    const NOMADReduction{M<:AbstractNOMAD,S} = NeuralOpReduction{M,S}

A reduction wrapper for the NOMAD (Non-linear Manifold Decoder) neural operator strategy.
It instructs the ROM solvers to use the NOMAD pipeline during the offline and online phases.

# Constructors
- `NOMADReduction(s::NeuralOpStrategy)`: Wraps an explicitly defined strategy.
- `NOMADReduction(; model = AutoNOMAD(), kwargs...)`: Automatically builds the strategy forwarding the keyword arguments.

# Examples
```julia
# Using an explicit strategy
strategy = NeuralOpStrategy(model=AutoNOMAD(), epochs=1000)
reduction = NOMADReduction(strategy)

# Using kwargs directly
reduction = NOMADReduction(model=AutoNOMAD(width=32, depth=2), epochs=1000)
```
"""
const NOMADReduction{M<:AbstractNOMAD,S} = NeuralOpReduction{M,S}

NOMADReduction(s::NeuralOpStrategy{<:AbstractNOMAD}) = NeuralOpReduction(s)
function NOMADReduction(;model = AutoNOMAD(),kwargs...)
  NeuralOpReduction(NeuralOpStrategy(;model=model,kwargs...))
end

"""
    const NeuralOpSolver{A,C<:NeuralOpReduction} = RBSteady.GlobalRBSolver{A,C,Nothing,Nothing}

    NeuralOpSolver(fesolver::GridapType, reduction::NeuralOpReduction)

Initializes the Reduced Basis Solver for Neural Operators.

# Arguments
- `fesolver`: The high-fidelity standard Gridap solver (e.g., `LUSolver()`). In the context of Reduced Order Models, the neural operator acts as a surrogate for this specific full-order solver. This reference defines the underlying high-fidelity model being approximated.
- `reduction::NeuralOpReduction`: The configured neural reduction strategy (e.g., `DeepONetReduction` or `NOMADReduction`).

# Examples

**Minimal Default Initialization:**
```julia
# Uses AutoDeepONet with default hyperparameters (20000 epochs, full-batch, etc.)
solver = NeuralOpSolver(LUSolver(), DeepONetReduction())
```
**Custom Initialization:**
```julia
using Lux

strategy = NeuralOpStrategy(
  model = AutoDeepONet(width=128, depth=4, activation=Lux.gelu),
  epochs = 1000
)
reduction = DeepONetReduction(strategy)
solver = NeuralOpSolver(ThetaMethod(LUSolver(), dt, θ), reduction)
```
"""
const NeuralOpSolver{A,C<:NeuralOpReduction} = RBSteady.GlobalRBSolver{A,C,Nothing,Nothing}

function NeuralOpSolver(fesolver,reduction::NeuralOpReduction)
  RBSolver(fesolver,RBSteady.GlobalContext(),reduction,nothing,nothing)
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
