abstract type AbstractTransientNNHyperReduction{A} <: TransientHyperReduction{A} end

"""
    struct TransientNNOperatorReduction <: AbstractTransientNNHyperReduction{NoReductionStyle}

Transient counterpart of [`NNOperatorReduction`](@ref). The NN maps parameter
values to the time-combined Galerkin-projected Jacobian, bypassing FE assembly.
Carry the `combination::TimeCombination` from the ODE solver.
"""
struct TransientNNOperatorReduction <: AbstractTransientNNHyperReduction{NoReductionStyle}
  combination::TimeCombination
  nparams::Int
  strategy::NeuralStrategy
end

function TransientNNOperatorReduction(
  combination::TimeCombination,
  args...;
  nparams::Int=20,
  model::NeuralNetwork=MultiLayerPerceptron(),
  strategy::NeuralStrategy=NeuralStrategy(model),
  kwargs...
  )

  TransientNNOperatorReduction(combination,nparams,strategy)
end

ParamDataStructures.num_params(r::TransientNNOperatorReduction) = r.nparams
get_strategy(r::TransientNNOperatorReduction) = r.strategy
RBTransient.get_time_combination(r::TransientNNOperatorReduction) = r.combination

"""
    struct TransientNNHyperReduction{A} <: AbstractTransientNNHyperReduction{A}

Transient counterpart of [`NNHyperReduction`](@ref). The NN predicts EIM
coefficients from parameter values; the time combination is applied at the
basis-projection stage.
"""
struct TransientNNHyperReduction{A} <: AbstractTransientNNHyperReduction{A}
  combination::TimeCombination
  reduction::Reduction{A,EuclideanNorm}
  strategy::NeuralStrategy
end

function TransientNNHyperReduction(
  combination::TimeCombination,
  args...;
  model::NeuralNetwork=MultiLayerPerceptron(),
  strategy::NeuralStrategy=NeuralStrategy(model),
  kwargs...
  )

  reduction = Reduction(args...;kwargs...)
  TransientNNHyperReduction(combination,reduction,strategy)
end

RBSteady.get_reduction(r::TransientNNHyperReduction) = r.reduction
get_strategy(r::TransientNNHyperReduction) = r.strategy
RBTransient.get_time_combination(r::TransientNNHyperReduction) = r.combination
