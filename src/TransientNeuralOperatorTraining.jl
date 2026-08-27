function train_neural_operator(
  red::DeepONetReduction,
  feop::ODEParamOperator,
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
  feop::ODEParamOperator,
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
  feop::ODEParamOperator,
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
  feop::ODEParamOperator,
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
