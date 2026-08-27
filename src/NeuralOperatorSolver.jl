"""
    reduced_operator(
      solver::NeuralOpSolver,
      feop::ParamOperator,
      s::AbstractSnapshots
    )

Executes the **Offline Phase** for Neural Operators on steady-state problems.
This method triggers the training loop of the neural network specified in the `solver`.

It automatically extracts the training dataset (parameters/sensors and spatial coordinates) from the snapshots `s` and the FE operator `feop`, normalizes the data, and performs the optimization.

Returns a `NeuralRBOperator` containing the trained network, its optimized weights, and the normalization statistics required for the online phase.
"""
function RBSteady.reduced_operator(
  solver::NeuralOpSolver,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  reduction = get_state_reduction(solver)
  model,ps,st,norm_stats,max_u = train_neural_operator(reduction,feop,s)
  NeuralRBOperator(feop,model,ps,st,norm_stats,max_u)
end

"""
    reduced_operator(
      solver::NeuralOpSolver,
      feop::ParamOperator,
      s::AbstractSnapshots,
      pretrained_op::NeuralRBOperator;
      update_stats::Bool = false
    )

Performs **Fine-Tuning (Continual or Transfer Learning)** on a previously trained Neural Operator.
It initializes the neural network with the weights and states of the `pretrained_op`, continuing the training using the newly provided snapshots `s` and the configuration defined in `solver`.

# Arguments
- `solver`: The `NeuralOpSolver` containing the updated training configuration (e.g., lower learning rate, new epochs).
- `feop`: The high-fidelity parametric operator.
- `s`: The new `Snapshots` dataset for fine-tuning.
- `pretrained_op`: The previously trained `NeuralRBOperator`.

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
solver_base = NeuralOpSolver(LUSolver(), DeepONetReduction(base_strategy))
pretrained_op = reduced_operator(solver_base, feop, snapshots_base)

# Fine-Tuning with a smaller learning rate on a refined dataset
ft_strategy = NeuralOpStrategy(
  model_arch, # match the pretrained one
  epochs = 1000,
  lr_scheduler = CosineAnnealing(1000, lr_max=1e-5) # Smaller LR
  )
solver_ft = NeuralOpSolver(LUSolver(), DeepONetReduction(ft_strategy))

# Continual learning (inherits original stats)
new_op = reduced_operator(solver_ft, feop, snapshots_new, pretrained_op; update_stats=false)
```
"""
function RBSteady.reduced_operator(
  solver::NeuralOpSolver,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
  )

  reduction = get_state_reduction(solver)
  model,ps,st,norm_stats,max_u = train_neural_operator(reduction,feop,s,pretrained_op;update_stats=update_stats)
  NeuralRBOperator(feop,model,ps,st,norm_stats,max_u)
end

"""
    reduced_operator(
      solver::NeuralOpSolver,
      s::AbstractSnapshots,
      pretrained_op::NeuralRBOperator;
      update_stats::Bool = false
    )

Automatically extracts the high-fidelity operator (`feop`) from `pretrained_op.op` and invokes the main fine-tuning routine.
"""
function RBSteady.reduced_operator(
  solver::NeuralOpSolver,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
  )

  feop = pretrained_op.op
  reduced_operator(solver,feop,s,pretrained_op;update_stats=update_stats)
end

function Algebra.solve(
  solver::NeuralOpSolver{A,<:DeepONetReduction},
  op::NeuralRBOperator,
  r::Realisation
  ) where A

  strategy = get_state_reduction(solver) |> get_strategy

  branch_stats = op.norm_stats.input
  trunk_stats = op.norm_stats.output

  # Branch Input (Parameters extraction)
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params,2)

  params_matrix = Float32.(sample(strategy.sampler.param_sampler,raw_params,2))
  normalise!(params_matrix,branch_stats)
  f_in = params_matrix

  # Trunk Input (Coordinates extraction, full resolution)
  V = get_test(op.op)

  x_test = coords_matrix(V)
  normalise!(x_test,trunk_stats)
  x_in = x_test

  # Inference Execution
  t = @timed begin
    pred_cpu,_ = op.model((f_in,x_in),op.model_weights,op.model_states)
  end

  # Denormalize output
  pred_cpu .*= op.max_u

  x̂ = Snapshots(ConsecutiveParamArray(pred_cpu),r)
  stats = CostTracker(t,nruns=n_samples,name="DeepONet Inference")

  return x̂,stats
end

function Algebra.solve(
  solver::NeuralOpSolver{A,<:NOMADReduction},
  op::NeuralRBOperator,
  r::Realisation
  ) where A

  nomad_net = op.model
  max_u = op.max_u
  strategy = get_state_reduction(solver) |> get_strategy

  u_in_stats = op.norm_stats.input
  y_in_stats = op.norm_stats.output

  # Parameters and Sensors extraction
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params,2)

  params_matrix = Float32.(sample(strategy.sampler.param_sampler,raw_params,2))
  n_sensors = size(params_matrix,1)

  # Coordinates extraction (full resolution)
  V = get_test(op.op)
  x_test = coords_matrix(V)
  D_phys = size(x_test,1)
  N_dofs = size(x_test,2)

  # Flattening of input tensors
  N_tot = N_dofs * n_samples
  u_in = zeros(Float32,n_sensors,N_tot)
  y_in = zeros(Float32,D_phys,N_tot)

  col = 1
  @views for sample_idx in 1:n_samples
    sensor_vals = params_matrix[:,sample_idx]
    for x_idx in 1:N_dofs
      u_in[:,col] .= sensor_vals
      y_in[:,col] .= x_test[:,x_idx]
      col += 1
    end
  end

  # Normalization
  normalise!(u_in,u_in_stats)
  normalise!(y_in,y_in_stats)

  # Inference
  t = @timed begin
    pred_cpu,_ = nomad_net((u_in,y_in),op.model_weights,op.model_states)
  end

  # Denormalization
  pred_cpu .*= max_u

  # Reshaping of the output for GridapROMs (N_dofs,n_samples)
  pred_2d = zeros(Float64,N_dofs,n_samples)
  col = 1
  for sample_idx in 1:n_samples
    for x_idx in 1:N_dofs
      pred_2d[x_idx,sample_idx] = pred_cpu[1,col]
      col += 1
    end
  end

  x̂ = Snapshots(ConsecutiveParamArray(pred_2d),r)
  stats = CostTracker(t,nruns=n_samples,name="NOMAD Inference")

  return x̂,stats
end
