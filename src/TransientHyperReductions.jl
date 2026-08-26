function RBSteady.HRProjection(
  red::HighDimNNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  ϕ = get_basis(y)
  model = TrainedNeuralNetwork(get_strategy(red),r,ϕ)
  return NNOperator(model,test)
end

function RBSteady.HRProjection(
  red::HighDimNNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  r = get_realisation(s)
  A = GalerkinProjectable(s)
  y = galerkin_projection(test,A,trial,get_time_combination(red))
  ϕ = permutedims(get_basis(y),(1,3,2))
  model = TrainedNeuralNetwork(get_strategy(red),r,ϕ)
  return NNOperator(model,trial,test)
end

function RBSteady.HRProjection(
  red::HighDimNNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(
  red::HighDimNNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial,get_time_combination(red))
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

const HighDimNNProjection{A<:Projection} = HRProjection{A,<:AbstractHighDimNNHyperReduction}
const HighDimNNContribution = AffineContribution{<:HighDimNNProjection}
const TupOfHighDimNNContribution = Tuple{Vararg{HighDimNNContribution}}

function FESpaces.interpolate!(
  b̂::AbstractArray,
  coeff::Tuple,
  a::TupOfHighDimNNContribution,
  r::AbstractRealisation
  )

  fill!(b̂,zero(eltype(b̂)))
  for (ai,ci) in zip(a,coeff)
    for aval in get_contributions(ai)
      interpolate!(b̂,ci,aval,r)
    end
  end
  return b̂
end
