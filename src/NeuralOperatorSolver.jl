"""
    reduced_operator(
      solver::NeuralSolver,
      feop::ParamOperator,
      s::AbstractSnapshots
    )

Executes the **Offline Phase** for Neural Operators on steady-state problems.
This method triggers the training loop of the neural network specified in the `solver`.

It automatically extracts the training dataset (parameters/sensors and spatial coordinates) from the snapshots `s` and the FE operator `feop`, normalizes the data, and performs the optimization.

Returns a `NeuralOperator` containing the trained network, its optimized weights, and the normalization statistics required for the online phase.
"""
function RBSteady.reduced_operator(
  solver::NeuralSolver,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  reduction = get_state_reduction(solver)
  model,ps,st,norm_stats = train_neural_operator(reduction,feop,s)
  NeuralOperator(feop,model,ps,st,norm_stats)
end

"""
    reduced_operator(
      solver::NeuralSolver,
      feop::ParamOperator,
      s::AbstractSnapshots,
      pretrained_op::NeuralOperator;
      update_stats::Bool=false
    )

Performs **Fine-Tuning (Continual or Transfer Learning)** on a previously trained Neural Operator.
It initializes the neural network with the weights and states of the `pretrained_op`, continuing the training using the newly provided snapshots `s` and the configuration defined in `solver`.

# Arguments
- `solver`: The `NeuralSolver` containing the updated training configuration (e.g., lower learning rate, new epochs).
- `feop`: The high-fidelity parametric operator.
- `s`: The new `Snapshots` dataset for fine-tuning.
- `pretrained_op`: The previously trained `NeuralOperator`.

# Keyword Arguments
- `update_stats::Bool`: Dictates how data normalization is handled.
  - If `false` (default): The model inherits the original normalization statistics (\$\\mu\$, \$\\sigma\$, and `max_u`) from the `pretrained_op`. Best for **Continual Learning** where the new data is drawn from the same underlying distribution.
  - If `true`: The model recomputes entirely new normalization statistics based solely on the new snapshots `s`. Best for **Transfer Learning** when shifting to a drastically different parameter space or domain scale.

# Examples
```julia
# Define the shared architecture (2 params -> Branch; 2D coords -> Trunk)
model_arch = DeepONet(2,2)

# Base Training
base_strategy = NeuralOpStrategy(model_arch, epochs=5000)
solver_base = NeuralSolver(LUSolver(), DeepONetReduction(base_strategy))
pretrained_op = reduced_operator(solver_base, feop, snapshots_base)

# Fine-Tuning with a smaller learning rate on a refined dataset
ft_strategy = NeuralOpStrategy(
  model_arch, # match the pretrained one
  epochs = 1000,
  lr_scheduler = CosineAnnealing(1000, lr_max=1e-5) # Smaller LR
  )
solver_ft = NeuralSolver(LUSolver(), DeepONetReduction(ft_strategy))

# Continual learning (inherits original stats)
new_op = reduced_operator(solver_ft, feop, snapshots_new, pretrained_op; update_stats=false)
```
"""
function RBSteady.reduced_operator(
  solver::NeuralSolver,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  reduction = get_state_reduction(solver)
  model,ps,st,norm_stats = train_neural_operator(reduction,feop,s,pretrained_op;update_stats=update_stats)
  NeuralOperator(feop,model,ps,st,norm_stats)
end

"""
    reduced_operator(
      solver::NeuralSolver,
      s::AbstractSnapshots,
      pretrained_op::NeuralOperator;
      update_stats::Bool=false
    )

Automatically extracts the high-fidelity operator (`feop`) from `pretrained_op.op` and invokes the main fine-tuning routine.
"""
function RBSteady.reduced_operator(
  solver::NeuralSolver,
  s::AbstractSnapshots,
  pretrained_op::NeuralOperator;
  update_stats::Bool=false
  )

  feop = pretrained_op.op
  reduced_operator(solver,feop,s,pretrained_op;update_stats=update_stats)
end

function Algebra.solve(
  solver::NeuralSolver{A,<:DeepONetReduction},
  op::NeuralOperator,
  r::Realisation
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  coords = get_coords(get_test(op.op))
  input = InputData(r,coords)
  input = sample(get_sampler(strategy).param_sampler,input)
  params,coords = get_formatted_data(Float32,input)
  normalise!((params,coords),op.norm_stats)

  # Inference Execution (denormalizes the output internally, using op.norm_stats.dmax)
  t = @timed begin
    pred_cpu,_ = op.model((params,coords),op.model_weights,op.model_states,op.norm_stats)
  end

  x̂ = Snapshots(ConsecutiveParamArray(pred_cpu),r)
  stats = CostTracker(t,nruns=num_params(r),name="DeepONet Inference")

  return x̂,stats
end

function Algebra.solve(
  solver::NeuralSolver{A,<:NOMADReduction},
  op::NeuralOperator,
  r::Realisation
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  coords = get_coords(get_test(op.op))
  input = InputData(r,coords)
  input = sample(get_sampler(strategy).param_sampler,input)
  params,coords = get_formatted_data(Float32,input)
  pin,xin = _flatten(params,coords)
  normalise!((pin,xin),op.norm_stats)

  # Inference (denormalizes the output internally, using op.norm_stats.dmax)
  t = @timed begin
    pred_cpu,_ = op.model((pin,xin),op.model_weights,op.model_states,op.norm_stats)
  end

  # Reshaping of the output for GridapROMs (N_dofs,n_samples)
  pred_2d = reshape(pred_cpu,size(coords,2),size(params,2))

  x̂ = Snapshots(ConsecutiveParamArray(pred_2d),r)
  stats = CostTracker(t,nruns=num_params(r),name="NOMAD Inference")

  return x̂,stats
end
