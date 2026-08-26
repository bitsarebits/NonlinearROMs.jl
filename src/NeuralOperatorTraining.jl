# Helpers and Devices

const CDEV = Lux.cpu_device()
const XDEV = Lux.reactant_device(;force=true)

function resolve_batch_size(batch_config::Int,total_samples::Int)
  return batch_config <= 0 ? total_samples : min(batch_config,total_samples)
end

function compute_zscore_stats(data::AbstractMatrix;normalise=false)
  if normalise 
    stats = compute_zscore_stats(data;normalise=false)
    normalise!(data,stats)
    return stats
  end
  μ = mean(data,dims=2)
  σ = std(data,dims=2)
  # Avoid dividing by zero if a feature is constant
  for i in eachindex(σ)
    iszero(σ[i]) && (σ[i] = one(eltype(σ)))
  end
  return (μ=Float32.(μ),σ=Float32.(σ))
end

normalise!(args...) = @abstractmethod

function normalise!(data::AbstractVector,law::NamedTuple)
  data .-= law.μ
  data ./= law.σ
  data
end

function normalise!(data::AbstractMatrix,law::NamedTuple)
  for v in eachcol(data)
    normalise!(v,law)
  end
  data 
end

"""
    get_coords(V::SingleFieldFESpace) -> Array{Point{D,Float64}}

Extracts the physical coordinates of the free DoFs of `V`, indexed consistently
with `V`'s own free-dof numbering (the same numbering used by
`get_all_data`/`get_free_dof_values` on `V`). Since the mapping is obtained by
directly interpolating the coordinate field onto `V`, it is valid for any
`SingleFieldFESpace` -- no special DoF ordering is required. Use
`sample(sampler::NeuralSampler,get_coords(V))` to subsample and stack the
result into a `(D_phys,N_dofs)` `Matrix{Float32}`.
"""
function get_coords(V::SingleFieldFESpace)
  order = get_polynomial_orders(V)
  trian = get_triangulation(V)
  model = get_background_model(trian)
  get_coords(model,order)
end

function get_coords(model::CartesianDiscreteModel{D},orders::NTuple{D,Int}) where D 
  desc = get_cartesian_descriptor(model)
  cells = CartesianIndices(desc.partition)
  nodes = CartesianIndices(orders .* desc.partition .+ 1 .- desc.isperiodic)
  coords = Array{Point{D,Float64}}(undef,size(nodes))
  for cell in cells
    first_new_node = orders .* (Tuple(cell) .- 1) .+ 1
    nodes_range = map(enumerate(first_new_node)) do (i,ni)
      ni:(ni+orders[i])
    end
    for inode in Iterators.product(nodes_range...)
      _is_periodic_node(inode,nodes) && continue
      coords[inode...] = Point(ntuple(d -> desc.origin[d] + (inode[d]-1)*desc.sizes[d],Val{D}()))
    end
  end
  return coords
end

function _is_periodic_node(inode,nodes)
  try
    nodes[inode...]
    return false
  catch
    return true
  end
end

"""
    coords_matrix(V::SingleFieldFESpace) -> Matrix{Float32}

Full-resolution counterpart of `sample(sampler::NeuralSampler,get_coords(V))`: stacks
every DoF coordinate of `V` (no spatial subsampling) into a `(D_phys,N_dofs)` matrix.
Used at inference time, where predictions are required at every DoF regardless of the
spatial subsampling used during training.
"""
coords_matrix(V::SingleFieldFESpace) = Float32.(stack(p -> collect(p.data),vec(get_coords(V))))

# Build model

function build_lux_chain(layers::Tuple,activation)
  lux_layers = []
  for i in 1:(length(layers)-1)
    if i < length(layers) - 1
      push!(lux_layers,Lux.Dense(layers[i] => layers[i+1],activation))
    else
      # last layer (no activation)
      push!(lux_layers,Lux.Dense(layers[i] => layers[i+1]))
    end
  end
  Lux.Chain(lux_layers...)
end

# Create a DeepONet layers
function LuxDeepONet(branch_net,trunk_net)
  Lux.Chain(
    # Process inputs (u,y) independently,then matrix-multiply them
    Lux.Parallel(
      *;
      # Branch: process 'u' -> shape (Features,Batch)
      # then transpose (adjoint) -> shape (Batch,Features)
      branch = Lux.Chain(branch_net,Lux.WrappedFunction(adjoint)),

      # Trunk: process 'y' -> shape (Features,Points)
      trunk = trunk_net
    ),
    # The '*' gives (Batch,Points).
    # Final transpose (adjoint) -> target shape: (Points,Batch)
    Lux.WrappedFunction(adjoint)
  )
end

function build_model(model::DeepONet)
  branch_net = build_lux_chain(model.branch_layers,model.activation)
  trunk_net = build_lux_chain(model.trunk_layers,model.activation)
  LuxDeepONet(branch_net,trunk_net)
end

function LuxNOMAD(approximator_net,decoder_net)
  Lux.Chain(
    # Apply approximator to 'u',pass 'y' untouched,and concatenate them (vcat)
    Lux.Parallel(
      vcat;
      approximator = approximator_net,
      y_pass_through = Lux.NoOpLayer()
    ),
    # Pass the concatenated vector [approximator(u); y] to the decoder
    decoder_net
  )
end

function build_model(model::NOMAD)
  approximator_net = build_lux_chain(model.approximator_layers,model.activation)
  decoder_net = build_lux_chain(model.decoder_layers,model.activation)
  LuxNOMAD(approximator_net,decoder_net)
end

# Training loop

function train_deeponet!(train_state,dataloader,x_data_dev,lr_scheduler;logger::TrainingLog)
  init!(logger)

  Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH) do
    for epoch = 1:logger.max_epochs
      local current_loss = 0.0f0

      for (f_batch,u_batch) in dataloader
        batch_dev = ((f_batch |> XDEV,x_data_dev),u_batch |> XDEV)

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

function train_nomad!(train_state,dataloader,lr_scheduler;logger::TrainingLog)
  init!(logger)

  Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH) do
    for epoch in 1:logger.max_epochs
      local current_loss = 0.0f0

      for ((u_batch,y_batch),v_batch) in dataloader
        # Single concatenated tensor (Sensors + Coordinates)
        batch_dev = (
          (u_batch |> XDEV,y_batch |> XDEV),
          v_batch |> XDEV
        )

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

# Generic Dispatch (Steady)

function train_neural_operator(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  strategy = red.strategy

  # Data extraction
  # RBSteady => get_all_data(s) is 2D: (N_dofs,N_samples)
  target_data_full = Float32.(get_all_data(s))
  N_dofs = size(target_data_full,1)

  idx_x = get_space_ids(strategy.sampler,N_dofs)
  target_data = @views target_data_full[idx_x,:]

  r = get_realisation(s)
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params,2)

  params_matrix = Float32.(sample(strategy.sampler.param_sampler,raw_params,2))

  # normalization
  max_u = maximum(abs,target_data)
  target_data ./= max_u

  # DoF coordinates extraction (Trunk input)
  V = get_test(feop)
  x_train = sample(strategy.sampler,get_coords(V)) # shape: (D_phys,N_dofs_reduced)

  # Input normalization
  branch_stats = compute_zscore_stats(params_matrix;normalise=true)
  trunk_stats = compute_zscore_stats(x_train;normalise=true)

  # Building the DeepONet
  deepONet = build_model(strategy.model)

  # Dataloader and setup
  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader(
    (params_matrix,target_data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  x_data_dev = x_train |> XDEV

  Random.seed!(42)
  ps,st = Lux.setup(Random.default_rng(),deepONet) |> XDEV

  train_state = Lux.Training.TrainState(deepONet,ps,st,strategy.optimiser.opt)

  # Executing the pipeline
  ps_trained,st_trained = train_deeponet!(
    train_state,dataloader,x_data_dev,strategy.optimiser.lr_scheduler;logger=strategy.trainlog
  )

  st_test = Lux.testmode(st_trained) |> CDEV

  norm_stats = (branch = branch_stats,trunk = trunk_stats)

  return deepONet,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end

function train_neural_operator(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
  )

  strategy = red.strategy

  # Data extraction
  # RBSteady => get_all_data(s) is 2D: (N_dofs,N_samples)
  target_data_full = Float32.(get_all_data(s))
  N_dofs = size(target_data_full,1)

  idx_x = get_space_ids(strategy.sampler,N_dofs)
  target_data = @views target_data_full[idx_x,:]

  r = get_realisation(s)
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params,2)

  params_matrix = Float32.(sample(strategy.sampler.param_sampler,raw_params,2))

  V = get_test(feop)
  x_train = sample(strategy.sampler,get_coords(V))

  nbranch_in = size(params_matrix,1)
  ntrunk_in = size(x_train,1)

  # Dimensions check for fine-tuning
  expected_branch_in = length(pretrained_op.norm_stats.branch.μ)
  expected_trunk_in = length(pretrained_op.norm_stats.trunk.μ)
  @assert nbranch_in == expected_branch_in "Branch dimension mismatch: expected $expected_branch_in,got $nbranch_in. Check the parameter sampler."
  @assert ntrunk_in == expected_trunk_in "Trunk dimension mismatch: expected $expected_trunk_in,got $ntrunk_in."

  # Normalization setup
  if update_stats
    strategy.trainlog.verbose && @info "Recomputing the normalization statistics."
    max_u = maximum(abs,target_data)
    branch_stats = compute_zscore_stats(params_matrix)
    trunk_stats = compute_zscore_stats(x_train)
  else
    strategy.trainlog.verbose && @info "Inheriting the normalization statistics from the pre-trained model."
    max_u = pretrained_op.max_u
    branch_stats = pretrained_op.norm_stats.branch
    trunk_stats = pretrained_op.norm_stats.trunk
  end

  # Normalization
  target_data ./= max_u
  normalise!(params_matrix,branch_stats)
  normalise!(x_train,trunk_stats)

  # Pretrained-model
  deepONet = pretrained_op.model
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV

  # Dataloader and optimization setup
  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader(
    (params_matrix,target_data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  x_data_dev = x_train |> XDEV

  train_state = Lux.Training.TrainState(deepONet,ps,st,strategy.optimiser.opt)

  # Training
  ps_trained,st_trained = train_deeponet!(
    train_state,dataloader,x_data_dev,strategy.optimiser.lr_scheduler;logger=strategy.trainlog
  )

  st_test = Lux.testmode(st_trained) |> CDEV
  norm_stats = (branch = branch_stats,trunk = trunk_stats)

  return deepONet,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end

function train_neural_operator(
  red::NOMADReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  strategy = red.strategy

  # Data extraction
  # RBSteady => get_all_data(s) is 2D: (N_dofs,N_samples)
  target_data_full = Float32.(get_all_data(s))
  N_dofs = size(target_data_full,1)

  idx_x = get_space_ids(strategy.sampler,N_dofs)
  N_x_red = length(idx_x)

  r = get_realisation(s)
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params,2)

  # Sensors extraction (like branch input in DeepONet)
  params_matrix = Float32.(sample(strategy.sampler.param_sampler,raw_params,2))
  n_sensors = size(params_matrix,1)

  # DoF coordinates extraction (like trunk input in DeepONet)
  V = get_test(feop)
  x_red = sample(strategy.sampler,get_coords(V)) # shape: (D_phys,N_x_red)
  D_phys = size(x_red,1)

  # Flattening for NOMAD
  N_tot = N_x_red * n_samples

  u_in = zeros(Float32,n_sensors,N_tot)
  y_in = zeros(Float32,D_phys,N_tot)
  v_out = zeros(Float32,1,N_tot)

  col_idx = 1
  @views for i in 1:n_samples
    sensor_vals = params_matrix[:,i]
    for (x_idx_reduced,x_idx_full) in enumerate(idx_x)
      u_in[:,col_idx] .= sensor_vals
      y_in[:,col_idx] .= x_red[:,x_idx_reduced]
      v_out[1,col_idx] = target_data_full[x_idx_full,i]
      col_idx += 1
    end
  end

  # normalization
  max_u = maximum(abs,v_out)
  v_out ./= max_u

  u_in_stats = compute_zscore_stats(u_in;normalise=true)
  y_in_stats = compute_zscore_stats(y_in;normalise=true)

  # Building the NOMAD model
  nomad_net = build_model(strategy.model)

  # DataLoader and Lux setup
  bs = resolve_batch_size(strategy.batch_size,N_tot)
  dataloader = MLUtils.DataLoader(
    ((u_in,y_in),v_out);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  Random.seed!(42)
  ps,st = Lux.setup(Random.default_rng(),nomad_net) |> XDEV

  train_state = Lux.Training.TrainState(nomad_net,ps,st,strategy.optimiser.opt)

  # Running the pipeline
  ps_trained,st_trained = train_nomad!(
    train_state,dataloader,strategy.optimiser.lr_scheduler;logger=strategy.trainlog
  )

  st_test = Lux.testmode(st_trained) |> CDEV

  # norm_stats
  norm_stats = (u_in = u_in_stats,y_in = y_in_stats)

  return nomad_net,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end

function train_neural_operator(
  red::NOMADReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
  )

  strategy = red.strategy

  # Data extraction
  target_data_full = Float32.(get_all_data(s))
  N_dofs = size(target_data_full,1)

  idx_x = get_space_ids(strategy.sampler,N_dofs)
  N_x_red = length(idx_x)

  r = get_realisation(s)
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params,2)

  params_matrix = Float32.(sample(strategy.sampler.param_sampler,raw_params,2))
  n_sensors = size(params_matrix,1)

  V = get_test(feop)
  x_red = sample(strategy.sampler,get_coords(V))
  D_phys = size(x_red,1)

  # Flattening for NOMAD
  N_tot = N_x_red * n_samples

  u_in = zeros(Float32,n_sensors,N_tot)
  y_in = zeros(Float32,D_phys,N_tot)
  v_out = zeros(Float32,1,N_tot)

  col_idx = 1
  @views for i in 1:n_samples
    sensor_vals = params_matrix[:,i]
    for (x_idx_reduced,x_idx_full) in enumerate(idx_x)
      u_in[:,col_idx] .= sensor_vals
      y_in[:,col_idx] .= x_red[:,x_idx_reduced]
      v_out[1,col_idx] = target_data_full[x_idx_full,i]
      col_idx += 1
    end
  end

  # Dimensions check for fine-tuning
  expected_u_in = length(pretrained_op.norm_stats.u_in.μ)
  expected_y_in = length(pretrained_op.norm_stats.y_in.μ)
  @assert n_sensors == expected_u_in "Sensors input dimension mismatch: expected $expected_u_in, got $n_sensors."
  @assert D_phys == expected_y_in "Coords input dimension mismatch: expected $expected_y_in, got $D_phys."

  # Normalization
  if update_stats
    strategy.trainlog.verbose && @info "Recomputing the normalization statistics."
    max_u = maximum(abs,v_out)
    u_in_stats = compute_zscore_stats(u_in)
    y_in_stats = compute_zscore_stats(y_in)
  else
    strategy.trainlog.verbose && @info "Inheriting the normalization statistics from the pre-trained model."
    max_u = pretrained_op.max_u
    u_in_stats = pretrained_op.norm_stats.u_in
    y_in_stats = pretrained_op.norm_stats.y_in
  end

  v_out ./= max_u
  normalise!(u_in,u_in_stats)
  normalise!(y_in,y_in_stats)

  # Pretrained model
  nomad_net = pretrained_op.model
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV

  # Dataloader and optimization setup
  bs = resolve_batch_size(strategy.batch_size,N_tot)
  dataloader = MLUtils.DataLoader(
    ((u_in,y_in),v_out);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  train_state = Lux.Training.TrainState(nomad_net,ps,st,strategy.optimiser.opt)

  # Training
  ps_trained,st_trained = train_nomad!(
    train_state,dataloader,strategy.optimiser.lr_scheduler;logger=strategy.trainlog
  )

  st_test = Lux.testmode(st_trained) |> CDEV
  norm_stats = (u_in = u_in_stats,y_in = y_in_stats)

  return nomad_net,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end
