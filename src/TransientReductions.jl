abstract type AbstractHighDimNNHyperReduction{A} <: HighDimHyperReduction{A} end

"""
    struct HighDimNNOperatorReduction <: AbstractHighDimNNHyperReduction{NoReductionStyle}

Transient counterpart of [`NNOperatorReduction`](@ref). The NN maps parameter
values to the time-combined Galerkin-projected Jacobian, bypassing FE assembly.
Carry the `combination::TimeCombination` from the ODE solver.
"""
struct HighDimNNOperatorReduction <: AbstractHighDimNNHyperReduction{NoReductionStyle}
  combination::TimeCombination
  nparams::Int
  strategy::NeuralStrategy
end

function HighDimNNOperatorReduction(
  combination::TimeCombination,
  args...;
  nparams::Int=20,
  model::NeuralNetwork=MultiLayerPerceptron(),
  strategy::NeuralStrategy=NeuralStrategy(model),
  kwargs...
  )

  HighDimNNOperatorReduction(combination,nparams,strategy)
end

ParamDataStructures.num_params(r::HighDimNNOperatorReduction) = r.nparams
get_strategy(r::HighDimNNOperatorReduction) = r.strategy
RBTransient.get_time_combination(r::HighDimNNOperatorReduction) = r.combination

"""
    struct HighDimNNHyperReduction{A} <: AbstractHighDimNNHyperReduction{A}

Transient counterpart of [`NNHyperReduction`](@ref). The NN predicts EIM
coefficients from parameter values; the time combination is applied at the
basis-projection stage.
"""
struct HighDimNNHyperReduction{A} <: AbstractHighDimNNHyperReduction{A}
  combination::TimeCombination
  reduction::Reduction{A,EuclideanNorm}
  strategy::NeuralStrategy
end

function HighDimNNHyperReduction(
  combination::TimeCombination,
  args...;
  model::NeuralNetwork=MultiLayerPerceptron(),
  strategy::NeuralStrategy=NeuralStrategy(model),
  kwargs...
  )

  reduction = Reduction(args...;kwargs...)
  HighDimNNHyperReduction(combination,reduction,strategy)
end

RBSteady.get_reduction(r::HighDimNNHyperReduction) = r.reduction
get_strategy(r::HighDimNNHyperReduction) = r.strategy
RBTransient.get_time_combination(r::HighDimNNHyperReduction) = r.combination
