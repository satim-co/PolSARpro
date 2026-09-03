# Oh surface inversion: C/Python parity

The legacy C implementation has an OpenMP race in its Newton iteration. The
loop counter `tt` is shared between workers, so a multithreaded run can execute
a varying number of the intended 100 iterations and produce different results
between identical runs.

Use a single C thread when collecting parity evidence:

```python
os.environ["OMP_NUM_THREADS"] = "1"
```

This must be set before running `surface_inversion_oh.exe`. Both comparison
notebooks set it explicitly.

## Reproducing the comparison

Run these notebooks from top to bottom:

1. `test-oh-surface-inversion-real-data-uavsar.ipynb` runs the normal Python
   implementation and compares it with single-threaded C.
2. `test-oh-surface-inversion-c-semantics-real-data-uavsar.ipynb` repeats the
   comparison with `c_semantics=True`.

The notebooks use the same UAVSAR product, thresholds, incidence angles, and C
output directory. They differ only in the Python output file and the
`c_semantics=True` argument.

For the normal Python implementation, the deterministic comparison reports
RMSE values of approximately `3.08e-6` for `ks`, `1.16e-6` for `er`, and
`2.39e-6` for `mv`. Its input mask is identical to C, while 878 output-mask
pixels differ because Python rejects non-finite estimates that C accepts.

With `c_semantics=True`, all reported mean and absolute errors are zero for the
three parameters and all masks. The metrics helper displays `1e-15` for RMSE
as its numerical floor; this does not represent an observed difference. A
direct comparison of the generated float32 arrays found all six outputs
bitwise identical to single-threaded C, including NaN locations and payloads.

The C-semantics mode reproduces the original mixture of float storage and
double-precision math-function evaluation, casts at the C assignment points,
and uses the same comparison-based validity rules. It is an experimental
option pending a decision about whether exact legacy behavior, including its
NaN masking, should become the default.
