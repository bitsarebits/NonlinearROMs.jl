# NonlinearROMs.jl Release Notes

## Review of the Neural Operators PR

This branch is a review pass on top of Isaia Zollo's neural-operators PR (merged at
`d6f6c18`, "merged, starting review"). The goal of this note is to separate what
actually changed from what only moved: most of the *algorithmic* code Isaia wrote —
the Lux/Reactant/Enzyme training loop, the device-transfer pattern, the DeepONet/NOMAD
architectures themselves, the GridapROMs integration points — is untouched. What
changed is the data-preparation/configuration plumbing around it, done in response to
real bugs that plumbing kept causing.

### What stayed the same

- **The training loop itself.** `train_model!`'s core — `Lux.Training.single_train_step!`
  with `Lux.AutoEnzyme()`/`Lux.MSELoss()`, wrapped in
  `Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH)` — is
  exactly as Isaia wrote it.
- **Device transfer.** `CDEV = Lux.cpu_device()` / `XDEV = Lux.reactant_device(;force=true)`
  and where each is applied (device before training, CPU after) is unchanged.
- **The architectures.** `LuxDeepONet`/`LuxNOMAD` (the `Lux.Parallel`-based branch/trunk
  and approximator/decoder chains) are the same networks.
- **The scheduler math.** Cosine annealing and plateau-reduction compute the same
  learning rate schedules as before; only how they're stored/invoked changed (below).
- **The overall GridapROMs integration shape.** `RBSolver` / `reduced_operator` /
  `Algebra.solve` multiple-dispatch pattern is the same one Isaia set up; this package
  still plugs into `RBSteady`/`RBTransient` the same way.

### What changed, and why

**1. One `NeuralStrategy` for every neural-network use case, not two.**
Before, DeepONet/NOMAD had their own training configuration, while the MLP-based
hyper-reduction regressors (`NNOperatorReduction`, `NNHyperReduction`) had a completely
separate `NNStrategy` (`type=MLPType()`, `layers`, `lr`, `optimiser`, `loss`, `epochs`,
`weight_decay`, `batch_size`, `lr_schedule`, `patience`, `val_fraction`). Now a single
`NeuralStrategy{A<:NeuralNetwork}` (model + epochs + batch_size + sampler + optimiser +
training log) configures all of them; `NNOperatorReduction`/`NNHyperReduction` just take
`model::NeuralNetwork` and `strategy::NeuralStrategy`. Two incompatible config types for
the same underlying task ("train a neural network") meant every improvement — a new
sampler option, a new scheduler — had to be built twice or left inconsistent between the
two paths.

**2. One subsampling abstraction (`Sampler`/`MultiSampler`) instead of separate
steady/transient code.**
Selecting a subset of spatial DoFs, parameters, and (for transient problems) time steps
before training used to be separate, independently-written logic for the steady and
transient cases. It's now one `sample(...)` multiple-dispatch entry point, driven by a
`MultiSampler(space_sampler,param_sampler,time_sampler)` where each sub-sampler is an
`Integer` stride, a `Function` transform, an explicit index vector, or `nothing`/`identity`
(no subsampling). This is the single biggest reason the transient bugs below existed:
logic written once for steady and re-derived for transient drifts, and several of the
bugs found during this review were exactly that kind of drift. Unifying it means a fix
in one place now fixes both.

**3. `CoordinateSnapshots` + one `get_formatted_data` for all data shapes.**
`CoordinateSnapshots` bundles a `Snapshots`/`TransientSnapshots` with the FE space's
physical DoF coordinates; `get_formatted_data(::Type{T}, s)` is now the single place that
turns any snapshot flavor (steady, transient, coordinate-wrapped, or a bare realisation +
coordinates pair for inference) into the plain `(data,params[,coords])` matrices a Lux
dataloader needs — including building the flattened space-time trunk-input grid for
transient DeepONet/NOMAD (`_spacetime_coords`). Previously this grid was rebuilt with
hand-written nested loops separately in each of the transient training/solve functions;
one shared implementation removes that duplication (and the bugs it invited).
An earlier pass of this review introduced an `InputData` wrapper (realisation + full-
resolution coordinates, for inference only); it turned out to be unneeded indirection
once `get_formatted_data` could be overloaded directly on `(realisation, coords)`, so it
was removed again in favor of that simpler form.

**4. `NeuralOperator`: weights and states now live inside the trained model.**
Before, `NeuralOperator` carried `model` (a bare Lux chain), `model_weights`,
`model_states`, and `norm_stats` as four separate fields that every caller — four
training functions × steady/transient × fresh/fine-tune — had to thread through by hand,
with the normalisation max-scale (`max_u`) tracked as yet another separate value.
Now `TrainedModel` (chain + parameters + states, CPU-resident) is a single object, and
`NeuralOperator` just holds `op`, `model::TrainedModel`, and `metadata` (a `NormStats`
bundling the z-score stats *and* the max-scale together, or `nothing` for architectures
with no normalisation). Inference is one call: `op.model((params,coords), op.metadata)`,
which applies the network and denormalises in the same step — a no-op when
`metadata === nothing`. Fewer independently-threaded pieces means fewer places for one
of them to silently get left behind, which is exactly what happened before this review
(see bugs below).

**5. File reorganization: split along the seams the code was actually growing along.**
Isaia's PR put almost everything — architectures, training loop, strategy/reduction/
operator types, and the coordinate/sampler machinery — into one ~665-line
`NeuralNetworks.jl`, plus a separate `TrainingLogs.jl`. This review split it up:

  | File | Now holds |
  |---|---|
  | `NeuralNetworks.jl` | the `NeuralNetwork` abstract type only |
  | `NeuralModels.jl` | DeepONet/NOMAD/MLP/AutoEncoder/VAE/AutoDecoder architectures |
  | `NeuralModelsTraining.jl` | `TrainedModel`, `train_model!`, generic trained-network wrappers |
  | `NeuralReductions.jl` | `NeuralStrategy`/`NeuralReduction`/`DeepONetReduction`/`NOMADReduction` |
  | `NeuralSolvers.jl` / `TransientNeuralSolver.jl` | `NeuralOperator`, `reduced_operator`, `Algebra.solve` |
  | `NeuralTraining.jl` / `TransientNeuralTraining.jl` | the DeepONet/NOMAD `train(...)` pipelines |
  | `Samplers.jl` | `Sampler`/`MultiSampler` |
  | `NeuralLayers.jl` | `LatentCodeLayer`/`VAELayer` |
  | `Utils.jl` | absorbed `TrainingLogs.jl`, plus `ZscoreStats`/`NormStats`/`normalise!`/`CoordinateSnapshots`/`get_formatted_data` |

  This is purely organizational: no behavior changed, other than the load-order
  constraints Julia imposes when a type moves to a file that now `include`s too late for
  another file's function signature to reference it (hit and fixed a few times over the
  course of this review).

**6. Learning-rate schedulers: encapsulated instead of re-passed every call.**
`AbstractLRScheduler` → `LRScheduler`; `get_initial_lr` → `get_lr`. `CosineAnnealing` now
stores `total_epochs` as a field set once at construction, instead of requiring every
`step_scheduler!` call to pass it in. `ReduceLROnPlateau`'s mutable counters
(`wait`/`best_loss`/`current_lr`) became `Base.Ref`s on an immutable `struct` rather than
fields of a `mutable struct`. The cosine/plateau math itself is unchanged.

### Bugs found and fixed during this review

Most of these only surfaced once the transient train/solve pipeline was exercised
end-to-end for the first time, or once fine-tuning and normalisation were exercised
together — the corresponding tests existed before but had never actually passed.

- DeepONet fine-tuning was discarding the pretrained weights and re-initializing them
  from scratch instead of continuing training from `pretrained_op`.
- The 3-argument `normalise!` (used when fine-tuning inherits stats instead of
  recomputing them) mapped `params`/`coords` to the wrong `NormStats` fields.
- `time_step = nothing` (the documented default, meaning "no time subsampling") had no
  matching `get_time_ids` method, so any transient training run using default settings
  errored immediately.
- Transient `Snapshots` construction (`ConsecutiveParamArray` wrapping a `(dofs,params,
  times)` array) used the wrong memory layout in a few places; GridapROMs' transient
  `Snapshots` machinery expects a flat `(dofs, params*times)` matrix with params varying
  fastest, not a literal 3D array.
- Fine-tuning's dimension-mismatch checks (asserting sensor/branch dimensions match the
  pretrained model) had been dropped somewhere in the training rewrite; restored them so
  mismatched fine-tuning data fails with a clear `AssertionError` instead of a cryptic
  broadcast error inside `normalise!`.
- The fine-tuning entry point was named `retrain_operator` while every docstring and
  test called `reduced_operator`; without a matching
  `reduced_operator(solver,feop,s,pretrained_op;...)` method, those calls silently fell
  through to GridapROMs' generic snapshot-*generating* fallback, which doesn't apply to
  fine-tuning (snapshots are already given) at all.
- A stale test still expected the pre-`dropdims` `(n,1)`-matrix shape for
  `ZscoreStats.μ`/`.σ`; the type itself is (correctly) constrained to `AbstractVector`.
- *(Pre-existing, flagged but not fixed — outside this review's scope, and order-1
  elements never trigger it):* `get_coords` mis-places nodes for higher-order
  (`order > 1`) reference elements — it doesn't divide the cell size by the polynomial
  order when subdividing a cell.

### Testing

`test/NeuralOperators/{deeponet,nomad}_pipeline.jl` now exercise fresh training,
fine-tuning (both `update_stats=true` and `update_stats=false`), and dimension-mismatch
error paths for **both** steady and transient problems. The transient testsets existed
in Isaia's PR but had never actually passed; they do now.
