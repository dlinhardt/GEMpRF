# Changelog

## 0.2.0

**Stored estimates changed. Results from 0.1.x are not directly comparable.** Expect differences up
to 5e-5 on essentially every parameter of every vertex, and a flipped sigma sign on 2-7% of them.
Nothing in the fit itself moved: the grid search, the refinement and R2 are unchanged.

The version check is a hard stop (`sys.exit(1)` on mismatch), so every config file needs
`version="0.2.0"` before it will run.

### Output format

- **Estimates are no longer rounded.** The record builder rounded every value to 4 decimals, and the
  HDF5 writer unpacks the same records, so the precise output format was quantised to 1e-4 -- far
  coarser than the float32 it stores. Rounding now happens only in the JSON writer, which is a
  human-readable dump. (The builder was called `args2jsonEntry`, which is how the rounding ended up
  there; it is now `args2estimate_record`.)
- **sigma is stored as a magnitude.** The Gaussian is even in sigma, so nothing kept the refinement
  on the positive branch and negative pRF sizes reached the results file. Anything downstream that
  averaged or thresholded on sigma mis-handled those vertices.
- **sigma is compared as a magnitude in the grid fallback.** `|refined - coarse|` measured across the
  sign, inflating the distance by `2|sigma|`, so whether a refinement was rejected as "too far"
  depended on how large sigma happened to be -- identical pRFs were judged differently. Measured
  impact on real data: 13 of 185,472 vertices change outcome.

### Fits that used to run out of memory

The fit and the signal synthesis now run on a single GPU. Peak device memory on a 785k-point grid
dropped from roughly 3x to 1x the error matrix in the assembly phase, and the model-signal buffers
from eight copies of the grid to about four. All of this is bit-identical -- verified over 185,472
vertices across both the individual and the concatenated path.

- the per-batch error matrix stayed alive while the next batch's was built (a third full copy)
- with one model-signal chunk the error matrix was assembled into a second array and copied
- the per-GPU signal buffer was held at the stimulus' full frame count until every chunk was written
- raw model signals stayed resident alongside their orthonormalized counterparts
- the derivative orthogonalization built four full arrays where two suffice

Batch sizes chosen by `<batches auto="true">` therefore differ from previous runs, which is the one
knob that can flip a vertex's grid winner or its refined-vs-grid decision (~0.05% of vertices).

### Fixed

- `gem.run(config)` raised `TypeError: 'module' object is not callable` on the second call in a
  process: importing the `gem.run` subpackage shadowed the entry point.
- A config living inside its own results directory was moved away by the backup step and then copied
  from a path that no longer existed.
- `CUDA_VISIBLE_DEVICES` being unset raised `KeyError` on a multi-GPU machine.
- The single-GPU fallback silently dropped to one signal per kernel launch instead of saying so.

## 0.1.15 and earlier

See the git history.
