"""
    module NonlinearROMs

Neural-network-based hyper-reduction and nonlinear reduced-order modelling
components for [`GridapROMs.jl`](https://github.com/gridap/GridapROMs.jl).

This package was extracted out of `GridapROMs.RBSteady`/`GridapROMs.RBTransient`
into its own repository, and plugs back into them via multiple dispatch:

- **Neural network models** (`NeuralOperatorModels.jl`) — `DeepONet`, `NOMAD`,
  `MultiLayerPerceptron`, `AutoEncoder`, `VariationalAutoEncoder`, `AutoDecoder`,
  `GenericNeuralNetwork`; all trained through the same Lux/Reactant/Enzyme
  pipeline (`NeuralOpStrategy`, `train_model!`, `TrainedNeuralNetwork`).

- **Steady hyper-reduction** — `NNOperatorReduction` (operator regression),
  `NNHyperReduction` (NN-predicted EIM coefficients), `NNOperator`,
  `NNInterpolation`, extending `RBSteady.HRProjection`/`RBSteady.Interpolation`
  and the `Algebra.residual!`/`Algebra.jacobian!` dispatch for `GenericRBOperator`.

- **Transient hyper-reduction** — `HighDimNNOperatorReduction`,
  `HighDimNNHyperReduction`, the transient counterparts extending
  `RBTransient`'s space-time hyper-reduction machinery analogously.

Usage: construct a `NNHyperReduction`/`NNOperatorReduction`
(or their `HighDim*` transient counterparts) and pass it to `RBSolver` wherever
a steady/transient `HyperReduction` is expected, exactly as you would
`MDEIMHyperReduction` or `RBFHyperReduction`.
"""
module NonlinearROMs

using LinearAlgebra
using Random
using SparseArrays
using Statistics
using FillArrays
using ForwardDiff
using Optimisers
using Enzyme
using Lux
using MLUtils
using Reactant

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.Helpers
using Gridap.Polynomials
using Gridap.ReferenceFEs

using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.ParamODEs
using GridapROMs.ParamSteady
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.Utils

import GridapROMs.RBSteady:
  GlobalRBSolver,GlobalContext,get_reduction,get_state_reduction,get_interpolation,
  allocate_coefficient,allocate_hyper_reduction,allocate_hypred_cache

export NeuralNetwork
export GenericNeuralNetwork
include("NeuralNetworks.jl")

export LRScheduler
export CosineAnnealing
export ReduceLROnPlateau
export step_scheduler!
export get_lr
include("LRSchedulers.jl")

export TrainingLog
include("TrainingLogs.jl")

export CoordinateSnapshots
export get_coords
include("CoordinateSnapshots.jl")

export Sampler
export NeuralSampler
export sample
export get_ids
export get_param_ids
export get_time_ids
include("NeuralSamplers.jl")

export DeepONet
export NOMAD
export MultiLayerPerceptron
export AutoEncoder
export VariationalAutoEncoder
export AutoDecoder
include("NeuralOperatorModels.jl")

export NeuralOptimiser
export NeuralOpStrategy
export NeuralOpReduction
export DeepONetReduction
export NOMADReduction
export NeuralSolver
export NeuralOperator
include("NeuralOperatorReductions.jl")

export train_neural_operator
export train_model!
export train_deeponet!
export train_nomad!
export TrainedNeuralNetwork
export TrainedModel
export TrainedAutoEncoder
export TrainedAutoDecoder
export TrainedVAE
export encode
export decode
export infer_latent
export resolve_batch_size
export XDEV
export CDEV
export build_model
export ZscoreStats
export normalise!
include("NeuralOperatorTraining.jl")
include("TransientNeuralOperatorTraining.jl")

include("NeuralOperatorSolver.jl")
include("TransientNeuralOperatorSolver.jl")

export NNOperatorReduction
export NNHyperReduction
include("SteadyReductions.jl")

export NNHRProjection
export NNOperator
export NNContribution
include("SteadyHyperReductions.jl")

export NNInterpolation
include("SteadyInterpolations.jl")

include("SteadyReducedOperators.jl")

export HighDimNNOperatorReduction
export HighDimNNHyperReduction
include("TransientReductions.jl")

export HighDimNNProjection
export HighDimNNContribution
include("TransientHyperReductions.jl")

include("TransientInterpolations.jl")

include("TransientReducedOperators.jl")

end # module
