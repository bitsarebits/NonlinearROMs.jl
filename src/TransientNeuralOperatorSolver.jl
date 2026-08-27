# Mirrors the steady `Algebra.solve` methods in NeuralOperatorSolver.jl exactly. The only
# transient-specific work is undoing the space-time flattening (space fastest, time slowest,
# per `_spacetime_coords`/`get_formatted_data` in CoordinateSnapshots.jl) after inference:
# `reshape` recovers `(N_dofs,N_time,n_samples)` directly from that column order, and
# `permutedims` brings it to GridapROMs' `(N_dofs,n_samples,N_time)` convention.

function Algebra.solve(
  solver::NeuralSolver{A,<:DeepONetReduction},
  op::NeuralOperator,
  r::TransientRealisation,
  args...
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  V = get_test(op.op)
  coords0 = get_coords(V)
  input = InputData(r,coords0)
  input = sample(get_sampler(strategy).param_sampler,input)
  params,coords = get_formatted_data(Float32,input)
  normalise!((params,coords),op.norm_stats)

  N_dofs = length(coords0)
  N_time = length(get_times(r))
  n_samples = num_params(r)

  # Inference Execution (denormalizes the output internally, using op.norm_stats.dmax)
  t = @timed begin
    pred_cpu,_ = op.model((params,coords),op.model_weights,op.model_states,op.norm_stats)
  end

  # Undo the space-time flattening back to (N_dofs,n_samples,N_time)
  pred_3d = reshape(pred_cpu,N_dofs,N_time,n_samples)
  pred_3d = permutedims(pred_3d,(1,3,2))

  x̂ = Snapshots(ConsecutiveParamArray(pred_3d),VectorDofMap(N_dofs),r)
  stats = CostTracker(t,nruns=n_samples,name="DeepONet Transient Inference")

  return x̂,stats
end

function Algebra.solve(
  solver::NeuralSolver{A,<:NOMADReduction},
  op::NeuralOperator,
  r::TransientRealisation,
  args...
  ) where A

  # Prepare input
  red = get_state_reduction(solver)
  strategy = get_strategy(red)
  V = get_test(op.op)
  coords0 = get_coords(V)
  input = InputData(r,coords0)
  input = sample(get_sampler(strategy).param_sampler,input)
  params,coords = get_formatted_data(Float32,input)
  pin,xin = _flatten(params,coords)
  normalise!((pin,xin),op.norm_stats)

  N_dofs = length(coords0)
  N_time = length(get_times(r))
  n_samples = num_params(r)

  # Inference (denormalizes the output internally, using op.norm_stats.dmax)
  t = @timed begin
    pred_cpu,_ = op.model((pin,xin),op.model_weights,op.model_states,op.norm_stats)
  end

  # Undo the NOMAD flattening back to (N_dofs,n_samples,N_time)
  pred_3d = reshape(pred_cpu,N_dofs,N_time,n_samples)
  pred_3d = permutedims(pred_3d,(1,3,2))

  x̂ = Snapshots(ConsecutiveParamArray(pred_3d),VectorDofMap(N_dofs),r)
  stats = CostTracker(t,nruns=n_samples,name="NOMAD Transient Inference")

  return x̂,stats
end
