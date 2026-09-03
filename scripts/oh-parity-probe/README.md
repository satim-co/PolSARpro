# Legacy Oh numerical parity probe

This experiment investigates numerical differences between the original
C-PolSARpro Oh inversion and its vectorized Python translation. It focuses on
the intermediate variables `a`, `b`, `c`, and the Newton-Raphson variable `x`.
Thresholds, validity masks, and the final `er`, `mv`, and `ks` calculations are
intentionally excluded from the probe.

## Motivation

The initial full UAVSAR comparison contains 1,752,231 pixels. Its
input-validity masks agree exactly, confirming that C and Python use the same
raster orientation, incidence angles, C3 channels, and threshold order. That
multithreaded C run differed from Python on 990 output-validity pixels. Later
repetition showed that the legacy C executable's multithreaded output is not
deterministic, so those initial counts are useful for selecting difficult data
but are not used as the final parity reference.

Most finite parameter results are close. Across the full image, 99% of the
absolute differences are no greater than approximately:

| Parameter | 99th-percentile absolute difference |
| --- | ---: |
| Dielectric constant (`er`) | `4.77e-6` |
| Volumetric moisture (`mv`) | `9.54e-6` |
| Normalized roughness (`ks`) | `7.15e-7` |

A small number of unstable Newton trajectories and mask changes dominate the
RMSE and maximum differences.

## Selected data

The probe uses the 64 x 64 UAVSAR crop:

```text
y = 650:714
x = 1223:1287
```

In the complete inversion, this crop contains:

- 55 output-mask differences;
- 54 non-finite C results;
- maximum absolute differences of approximately 17.6 for `er`, 31.5 for `mv`,
  and 1.57 for `ks`.

Six synthetic records are appended to exercise NaN inputs, zero VV, and
`HH/VV = 1`.

## Compared implementations

`oh_probe.c` reproduces the original C expressions and writes four float
values per record: `a`, `b`, `c`, and `x`.

`oh_probe.py` provides three alternatives:

1. **float32** uses NumPy float32 arrays throughout. This represents the
   arithmetic behavior of the current vectorized implementation.
2. **C-semantics** stores C `float` variables as float32, evaluates expressions
   involving the C double-precision math functions in float64, and casts back
   to float32 at the corresponding C assignment points.
3. **float64** keeps both stored values and calculations in float64. It tests
   whether simply increasing all precision reproduces C; it does not.

`oh_probe_float.c` provides a fourth experiment. Unlike the original C source,
it uses only float constants and the float math functions `sqrtf`, `logf`, and
`expf`. It tests whether changing C itself to all-float arithmetic reproduces
the NumPy float32 path.

The comparison records `x` after 1, 2, 5, 10, 20, 50, and 100 iterations.

## Meaning of "different"

The report distinguishes frequency from magnitude. A large count in
`exact_diff` does not necessarily indicate a scientifically important error.

- `exact_diff` counts records for which `C_value == Python_value` is false.
  Any finite rounding difference counts, even one at the last float32 bit.
  Values that are NaN in both implementations are treated as matching. This
  is numeric equality rather than a comparison of NaN payload bits; positive
  and negative zero also compare equal.
- `nan_diff` counts records where exactly one implementation produces NaN.
- `finite_n` is the number of records for which both results are finite.
- `p99_abs` is the 99th percentile of `abs(C - Python)`, calculated only over
  records finite in both implementations.
- `max_abs` is the largest absolute difference over those finite records.

Consequently, `exact_diff` answers "how often is there any rounding
difference?", while `p99_abs` and `max_abs` describe how large the differences
are. `nan_diff` is reported separately because subtracting NaNs does not
produce a useful error magnitude.

For mask comparisons in the full inversion, "different" means one mask is 0
and the other is 1 at the same pixel.

## Results

Before the Newton loop, the ordinary float32 path already differs from C:

| Variable | Records numerically unequal | Maximum absolute difference |
| --- | ---: | ---: |
| `a` | 1,183 | `2.98e-8` |
| `b` | 910 | `1.19e-7` |
| `c` | 3,168 | `2.38e-7` |

These are frequent but very small rounding differences. The unstable Newton
iteration can amplify them:

| Implementation | Iteration | Unequal `x` | NaN-pattern differences | 99th-percentile absolute difference | Maximum finite difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 1 | 2,566 | 0 | `3.11e-5` | `2.14e-2` |
| float32 | 2 | 3,331 | 0 | `1.42e12` | `7.94e35` |
| float32 | 100 | 700 | 4 | `2.15e-6` | `0.660` |
| float64 | 100 | 1,264 | 5 | `1.23e-6` | `0.266` |
| C-semantics | every checkpoint | 0 | 0 | `0` | `0` |

The C-semantics Python probe matches every stored C value at every checkpoint.
This demonstrates that the important behavior is not simply float32 or
float64. It is the combination used by C:

1. store variables as `float`;
2. evaluate expressions containing `log`, `exp`, and related C math functions
   in `double`;
3. round back to `float` at each assignment, especially after every update of
   `x`.

The very large iteration-2 differences show that the fixed Newton procedure is
numerically unstable for some inputs. Some trajectories subsequently converge
or become non-finite, so the iteration-100 percentile alone does not describe
the intermediate instability.

## All-float C experiment

Changing the C probe to float constants and `sqrtf`/`logf`/`expf` makes its
initial variables exactly equal to the Python float32 values:

| Comparison | Variable | Numerically unequal records | Maximum absolute difference |
| --- | --- | ---: | ---: |
| Python float32 vs all-float C | `a` | 0 | `0` |
| Python float32 vs all-float C | `b` | 0 | `0` |
| Python float32 vs all-float C | `c` | 0 | `0` |

The Newton results are still not identical:

| Reference and candidate | Iteration | Unequal `x` | NaN-pattern differences | 99th-percentile absolute difference | Maximum finite difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original C vs all-float C | 1 | 2,545 | 0 | `2.74e-5` | `2.12e-2` |
| Original C vs all-float C | 100 | 685 | 3 | `2.15e-6` | `6.80e-2` |
| All-float C vs Python float32 | 1 | 873 | 0 | `1.91e-6` | `1.41e-4` |
| All-float C vs Python float32 | 100 | 285 | 3 | `2.38e-7` | `0.171` |

This separates two effects:

1. Using float constants and float math functions removes the initial `a`,
   `b`, and `c` differences.
2. C `logf`/`expf` and NumPy's vectorized float32 `log`/`exp` are not
   bit-identical on this system. Their small differences are amplified by the
   unstable Newton iteration.

The all-float C probe is closer to Python float32 in frequency and typical
error, but it is neither identical to Python nor representative of the
original PolSARpro implementation. The C-semantics Python path remains the
only tested variant with zero differences from the original C probe at every
checkpoint.

## Full-image C-semantics result

The C-semantics experiment was also implemented as a temporary vectorized
xarray/Dask core and evaluated on the complete UAVSAR image. The legacy C
executable produced different output hashes on repeated default-thread runs.
Its Newton counter `tt` is shared by all workers because it is missing from the
OpenMP `private(...)` clause, so threads race while deciding how many of the
100 iterations to execute. With `OMP_NUM_THREADS=1`, two complete runs produced
byte-identical hashes for all six output files, providing a stable reference
for both Python notebooks. The separate shared `ligDone` counters also race,
but they are used only for progress reporting.

Against that single-thread reference, the ordinary Python core has the
following exact differences:

| Output | Different pixels | NaN-pattern differences | Maximum finite absolute difference |
| --- | ---: | ---: | ---: |
| `oh_ks` | 517,549 | 878 | `2.78e-3` |
| `oh_er` | 453,254 | 878 | `4.87e-4` |
| `oh_mv` | 482,236 | 878 | `1.09e-3` |

After extending the experimental core to reproduce C's comparison-based NaN
mask behavior, all six full-image outputs are bitwise identical to the
single-thread C files. This includes `oh_ks`, `oh_er`, `oh_mv`, both masks, the
combined mask, and the locations and payloads of NaNs.

In the executed notebooks, the ordinary core took about 5.37 seconds and the
C-semantics core about 7.48 seconds on this system. These are development
measurements rather than formal benchmarks.

## NaN behavior

For explicit NaN inputs and zero VV, C and all three Python variants produce a
NaN `x`. The core `a`/`b`/`c`/`x` arithmetic therefore propagates these cases
consistently.

The later C mask logic is a separate issue. Conditions such as
`er_inv >= 20 || er_inv < 0` are both false for NaN, causing C to mark some
non-finite results as valid. In the full-image comparison, 869 of the 893
pixels accepted only by C contain non-finite C estimates. Python deliberately
rejects non-finite estimates.

## Running the comparison

From the repository root:

```bash
python scripts/oh-parity-probe/compare.py
```

The script compiles the C probe and writes all generated artifacts to:

```text
/tmp/polsarpro-oh-parity/
```

Only the two C probes, Python source, comparison script, and this report are
kept in the repository.
