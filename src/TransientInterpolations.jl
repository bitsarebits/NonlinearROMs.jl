function RBSteady.Interpolation(
  red::TransientNNHyperReduction,
  a::KroneckerProjection,
  s::TransientSnapshots
  )

  inds,interp = DEIM(a)
  factor = lu(interp)
  r = get_params(get_realisation(s))
  red_data = RBTransient.get_at_kron_domain(s,inds...)
  coeff = parameterise(allocate_in_domain(a),r)
  ldiv!(coeff,factor,red_data)
  model = train_neural_coefficient(get_strategy(red),r,coeff)
  NNInterpolation(model)
end

function RBSteady.Interpolation(
  red::TransientNNHyperReduction,
  a::SequentialProjection,
  s::TransientSnapshots
  )

  inds,interp = DEIM(a)
  factor = lu(interp)
  r = get_params(get_realisation(s))
  red_data = RBTransient.get_at_seq_domain(s,inds...)
  coeff = parameterise(allocate_in_domain(a),r)
  ldiv!(coeff,factor,red_data)
  model = train_neural_coefficient(get_strategy(red),r,coeff)
  NNInterpolation(model)
end
