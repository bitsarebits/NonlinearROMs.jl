# NonlinearROMs.jl Release Notes

## Review of the Neural Operators PR

This branch is a review pass on top of Isaia's neural-operators PR (merged at `d6f6c18`, "merged, starting review"). The goal of this note is to separate what actually changed from what only moved: most of the *algorithmic* code Isaia wrote — the Lux/Reactant/Enzyme training loop, the device-transfer pattern, the DeepONet/NOMAD architectures themselves, the GridapROMs integration points — is untouched. The changes that have been made aim to:
- Simplify the code by finding common structures/operations used in different parts of the library, and replacing the old code by use these new functions instead. This will improve extendibility for future releases.
- Integrate Isaia's architecture with my pre-existing neural networks. The result is that we have already developed a fairly mature suite of nonlinear models in our package. 

### What stayed the same

- **The training loop itself.** `train_model!`'s core — `Lux.Training.single_train_step!` with `Lux.AutoEnzyme()`/`Lux.MSELoss()`, wrapped in
  `Reactant.with_config(;dot_general_precision=Reactant.PrecisionConfig.HIGH)` — is exactly as Isaia wrote it.
- **Device transfer.** `CDEV = Lux.cpu_device()` / `XDEV = Lux.reactant_device(;force=true)` and where each is applied (device before training, CPU after) is unchanged.
- **The architectures.** `LuxDeepONet`/`LuxNOMAD` (the `Lux.Parallel`-based branch/trunk and approximator/decoder chains) are the same networks.
- **The scheduler math.** Cosine annealing and plateau-reduction compute the same learning rate schedules as before; only how they're stored/invoked changed (below).
- **The overall GridapROMs integration shape.** `RBSolver` / `reduced_operator` / `Algebra.solve` multiple-dispatch pattern is the same one Isaia set up; this package still plugs into `RBSteady`/`RBTransient` the same way.

### What changed, and why

**1. File reorganization: split along the seams the code was actually growing along.**
I changed several file names and introduced new files, so that the overall structure of the package is the following:

  | File | Now holds |
  |---|---|
  | `NeuralNetworks.jl` | the `NeuralNetwork` abstract type only |
  | `NeuralModels.jl` | DeepONet/NOMAD/MLP/AutoEncoder/VAE/AutoDecoder architectures |
  | `NeuralModelsTraining.jl` | `TrainedNeuralModel`, `train_model!`, generic trained-network wrappers |
  | `NeuralReductions.jl` | `NeuralStrategy`/`NeuralReduction`/`DeepONetReduction`/`NOMADReduction` |
  | `NeuralSolvers.jl` / `TransientNeuralSolver.jl` | `NeuralOperator`, `reduced_operator`, `Algebra.solve` |
  | `NeuralTraining.jl` / `TransientNeuralTraining.jl` | the DeepONet/NOMAD `train(...)` pipelines |
  | `Samplers.jl` | `Sampler`/`MultiSampler` |
  | `NeuralLayers.jl` | `LatentCodeLayer`/`VAELayer` |
  | `Utils.jl` | absorbed `TrainingLogs.jl`, plus `ZscoreStats`/`NormStats`/`normalise!`/`CoordinateSnapshots`/`get_formatted_data` |

This is purely organizational: no behavior changed, other than the load-order constraints Julia imposes when a type moves to a file that now `include`s too late for another file's function signature to reference it (hit and fixed a few times over the course of this review).

**2. One `NeuralStrategy` for every neural-network use case, not two.**
Before, DeepONet/NOMAD had their own training configuration, while the MLP-based hyper-reduction regressors (`NNOperatorReduction`, `NNHyperReduction`) had a completely separate `NNStrategy` (`type=MLPType()`, `layers`, `lr`, `optimiser`, `loss`, `epochs`, `weight_decay`, `batch_size`, `lr_schedule`, `patience`, `val_fraction`). Now a single
`NeuralStrategy{A<:NeuralNetwork}` (model + epochs + batch_size + sampler + optimiser + training log) configures all of them. Also, I removed the `AutoDeepONet`/`AutoNOMAD` structs, as these can easily be merged with the non-`Auto` versions. Consequently, `resolve_model` was also removed.

**3. One subsampling abstraction (`Sampler`/`MultiSampler`) instead of separate steady/transient code.**
Selecting a subset of spatial DoFs, parameters, and (for transient problems) time steps before training used to be separate, independently-written logic for the steady and transient cases. It's now one `sample(...)` multiple-dispatch entry point, driven by a
`MultiSampler(space_sampler,param_sampler,time_sampler)` where each sub-sampler is an `Integer` stride, a `Function` transform, an explicit index vector, or `nothing`/`identity` (no subsampling). This allows us to cleanly select the datasets for training and testing without having to rewrite every time several (cumbersome) `for` loops.

**4. Learning-rate schedulers: encapsulated instead of re-passed every call.**
`AbstractLRScheduler` → `LRScheduler`; `get_initial_lr` → `get_lr`. `CosineAnnealing` now stores `total_epochs` as a field set once at construction, instead of requiring every `step_scheduler!` call to pass it in. `ReduceLROnPlateau`'s mutable counters (`wait`/`best_loss`/`current_lr`) became `Base.Ref`s on an immutable `struct` rather than
fields of a `mutable struct`. The cosine/plateau math itself is unchanged.

**5. `NeuralOperator`: weights and states now live inside the trained model.**
Before, `NeuralOperator` carried `model` (a bare Lux chain), `model_weights`, `model_states`, `norm_stats` and `max_u` as separate fields. I see these fields as two macro-structures: 
- Chain + parameters + states: these make up a single `TrainedNeuralModel`.
- The rest can be seen as a structure collecting information on the normalisation factors of the data. By default, this structure is a `NormStats` which basically contains the old `norm_stats` and `max_u`; however, this could also be of type `Nothing`, if no normalisation is applied.
In essence, now a `NeuralOperator` contains only two fields: a `model <: TrainedNeuralModel`, and `metadata <: Union{NormStats, Nothing}`.

**6. Now using my proposed coordinates builder.**
`CoordinateSnapshots` bundles a `Snapshots`/`TransientSnapshots` together with the FE
space's physical DoF coordinates, and `get_formatted_data(::Type{T}, s)` is the single
place that turns any of them (steady, transient, or coordinate-wrapped) into the plain
`(data,params[,coords])` matrices a Lux dataloader needs — including building the
flattened space-time trunk-input grid for transient DeepONet/NOMAD (`_spacetime_coords`),
which used to be rebuilt by hand with nested loops separately in each transient
training/solve function. For inference, `get_formatted_data(::Type{T}, r, coords)` takes
the (possibly-sampled) realisation and the FE space's coordinates directly — an earlier
pass of this review had introduced an `InputData` wrapper struct for this (realisation +
full-resolution coordinates), but it turned out to be unneeded indirection once
`get_formatted_data` could just be overloaded on the pair directly, so it was removed
again.

**7. `AutoEncoder`/`AutoDecoder`/`VariationalAutoEncoder` now go through `reduced_operator` too.**
Originally these three were only reachable through `TrainedNeuralNetwork` (now
`train_neural_coefficient`, see below), which builds the chain and calls `train_model!`
inline — never producing a `NeuralOperator`. `MultiLayerPerceptron` genuinely belongs
there (it's what `NNOperatorReduction`/`NNHyperReduction` use to regress MDEIM
coefficients from parameters, nothing to do with the `NeuralOperator` surrogate role),
but AutoEncoder/AutoDecoder/VAE reconstruct snapshot data, which is exactly the kind of
thing worth training through the same `train(red,feop,s)`/`train(red,feop,s,pretrained_op)`
pipeline DeepONet/NOMAD use. So they now have their own `AutoEncoderReduction`/
`AutoDecoderReduction`/`VAEReduction` (the same `NeuralReduction{M}` alias pattern as
`DeepONetReduction`/`NOMADReduction`) and their own `train_autoencoder!`/`train_autodecoder!`/
`train_vae!` thin wrappers mirroring `train_deeponet!`/`train_nomad!` exactly. Since they
train on the snapshot data alone (reconstruction, data → data, no parameters/coordinates
involved), the resulting `NeuralOperator` carries `metadata = nothing` — there's no
normalisation stats to compute for them.

To let VAE's custom reconstruction+KL loss share the same loop, `train_model!` now takes
an optional `loss` (default `Lux.MSELoss()`), which also fixed something I hadn't
noticed until I looked: the old standalone VAE training loop never moved anything to
`XDEV`, so VAE was the only architecture training on the CPU instead of through
Reactant/XLA like everything else. It goes through the same path now.

`AutoEncoder`/`AutoDecoder`/`VariationalAutoEncoder` don't fix their input dimension at
construction time the way DeepONet/NOMAD do (branch/trunk sizes are given up front;
these need the snapshot dimensionality, known only once the data is in hand), so
`build_model` gained overloads taking that dimension explicitly:
`build_model(model::AutoEncoder, nin)`, `build_model(model::AutoDecoder, nin, n_train)`
(the number of training snapshots, since `AutoDecoder`'s latent codes are optimised
per-sample), `build_model(model::VariationalAutoEncoder, nin)`.

`AutoDecoder` fine-tuning keeps the pretrained decoder weights but fits fresh latent
codes for the new snapshot set, since codes are tied to specific training samples and
can't be inherited across a different one.

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
- `TrainedAutoDecoder`'s `Arrays.evaluate!`/`infer_latent` referenced a `.decoder` field
  and a flat `.parameters`/`.states` that don't exist on `TrainedNeuralModel` — leftover
  from before the chain+parameters+states unification; fixed to use
  `.chain.layers.layer_2`/`.parameters.layer_2`/`.states.layer_2`, matching
  `get_latent_codes`'s already-correct style. (`TrainedAutoEncoder`/`TrainedAutoDecoder`
  are still declared as `TrainedNeuralModel{<:AutoEncoder}`/`{<:AutoDecoder}`, which
  can't actually match anything — `TrainedNeuralModel`'s type parameter is the *Lux
  chain's* type, never the recipe struct — so `encode`/`decode`/`infer_latent` are still
  unreachable through them; fixing that needs a real design decision, since
  `NeuralOperator` requires `model::TrainedNeuralModel` and Julia can't subtype a
  concrete struct, so flagging it here rather than guessing.)
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
in Isaia's PR but had never actually passed; they do now. `AutoEncoderReduction`/
`AutoDecoderReduction`/`VAEReduction` are exercised by an ad hoc smoke test (fresh +
fine-tune, all three architectures) rather than a committed test file so far — worth
promoting into `test/NeuralOperators/` alongside the DeepONet/NOMAD pipelines.