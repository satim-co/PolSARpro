"""Compare C and Python Oh intermediate variables on a high-error crop."""

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

from oh_probe import probe_c_semantics, probe_float32, probe_float64
from polsarpro.io import open_netcdf_beam

SOURCE_DIR = Path(__file__).parent
ARTIFACT_DIR = Path("/tmp/polsarpro-oh-parity")
PRODUCT_ID = "winnip_09002_12060_004_120714_L090_CX_02"
INPUT_NETCDF = Path(f"/data/psp/test_files/{PRODUCT_ID}_ML5X5.nc")
ANGLE_FILE = Path("/data/psp/winnip_cpsp/incidence_angle_ml5x5_rad.bin")
CROP = {"y": slice(650, 714), "x": slice(1223, 1287)}
CHECKPOINTS = (0, 1, 2, 5, 10, 20, 50, 100)
FIELDS = ("a", "b", "c", "x")


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    c_executable = _compile_probe("oh_probe.c", "oh_probe")
    c_float_executable = _compile_probe("oh_probe_float.c", "oh_probe_float")
    inputs, labels = _load_inputs()
    input_file = ARTIFACT_DIR / "probe_input.bin"
    inputs.tofile(input_file)

    print(
        f"Crop: y={CROP['y'].start}:{CROP['y'].stop}, "
        f"x={CROP['x'].start}:{CROP['x'].stop}"
    )
    print(f"Records: {len(inputs)} ({len(inputs) - 6} real + 6 synthetic)")
    print("Thresholds and validity masks are intentionally omitted.\n")

    c_results = _run_c_probe(c_executable, input_file, "probe_output")
    c_float_results = _run_c_probe(c_float_executable, input_file, "probe_float_output")

    variants = {
        "float32": probe_float32,
        "C-semantics": probe_c_semantics,
        "float64": probe_float64,
    }

    print("Intermediate a/b/c comparison (calculated before Newton iteration)")
    print(_header())
    reference = c_results[0]
    for variant_name, function in variants.items():
        candidate = function(*inputs.T, iterations=0)
        for field_index, field in enumerate(FIELDS[:3]):
            print(
                _summary_row(
                    variant_name,
                    0,
                    field,
                    reference[:, field_index],
                    candidate[:, field_index],
                )
            )
    for field_index, field in enumerate(FIELDS[:3]):
        print(
            _summary_row(
                "C-all-float",
                0,
                field,
                reference[:, field_index],
                c_float_results[0][:, field_index],
            )
        )

    print("\nNewton x trajectory")
    print(_header())
    for iterations in CHECKPOINTS[1:]:
        reference = c_results[iterations][:, 3]
        for variant_name, function in variants.items():
            candidate = function(*inputs.T, iterations=iterations)[:, 3]
            print(_summary_row(variant_name, iterations, "x", reference, candidate))
        print(
            _summary_row(
                "C-all-float",
                iterations,
                "x",
                reference,
                c_float_results[iterations][:, 3],
            )
        )

    print("\nPython float32 versus all-float C")
    print(_header())
    python_float32 = probe_float32(*inputs.T, iterations=0)
    for field_index, field in enumerate(FIELDS[:3]):
        print(
            _summary_row(
                "float32",
                0,
                field,
                c_float_results[0][:, field_index],
                python_float32[:, field_index],
            )
        )
    for iterations in CHECKPOINTS[1:]:
        candidate = probe_float32(*inputs.T, iterations=iterations)[:, 3]
        print(
            _summary_row(
                "float32",
                iterations,
                "x",
                c_float_results[iterations][:, 3],
                candidate,
            )
        )

    print("\nLargest finite x differences after 100 iterations")
    reference = c_results[100][:, 3]
    for variant_name, function in variants.items():
        candidate = function(*inputs.T, iterations=100)[:, 3]
        finite = np.isfinite(reference) & np.isfinite(candidate)
        finite_indices = np.flatnonzero(finite)
        order = finite_indices[
            np.argsort(np.abs(reference[finite] - candidate[finite]))[-8:][::-1]
        ]
        print(f"\n{variant_name}")
        for index in order:
            print(
                f"  {labels[index]:>18}  C={reference[index]: .9g}  "
                f"Python={candidate[index]: .9g}  "
                f"abs_diff={abs(reference[index] - candidate[index]):.9g}"
            )

    print("\nSynthetic NaN/singularity cases after 100 iterations")
    synthetic_start = len(inputs) - 6
    for index in range(synthetic_start, len(inputs)):
        values = [f"{labels[index]:>18}", f"C={c_results[100][index, 3]!r}"]
        values.append(f"C-all-float={c_float_results[100][index, 3]!r}")
        for variant_name, function in variants.items():
            candidate = function(*inputs[index], iterations=100)[0, 3]
            values.append(f"{variant_name}={candidate!r}")
        print("  ".join(values))


def _compile_probe(source_name, executable_name):
    source = SOURCE_DIR / source_name
    executable = ARTIFACT_DIR / executable_name
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-O0",
            "-Wall",
            "-Wextra",
            str(source),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def _run_c_probe(executable, input_file, output_stem):
    results = {}
    for iterations in CHECKPOINTS:
        output_file = ARTIFACT_DIR / f"{output_stem}_{iterations:03d}.bin"
        subprocess.run(
            [executable, input_file, output_file, str(iterations)], check=True
        )
        results[iterations] = np.fromfile(output_file, dtype=np.float32).reshape(-1, 4)
    return results


def _load_inputs():
    dataset = open_netcdf_beam(INPUT_NETCDF).isel(CROP).compute()
    full_shape = (753, 2327)
    theta = np.memmap(ANGLE_FILE, dtype=np.float32, mode="r", shape=full_shape)[
        CROP["y"], CROP["x"]
    ]

    real_inputs = np.column_stack(
        (
            np.asarray(theta).ravel(),
            dataset.m11.values.ravel(),
            dataset.m33.values.ravel(),
            (dataset.m22.values / np.float32(2.0)).ravel(),
        )
    ).astype(np.float32)
    labels = [
        f"y={y},x={x}"
        for y in range(CROP["y"].start, CROP["y"].stop)
        for x in range(CROP["x"].start, CROP["x"].stop)
    ]

    synthetic_inputs = np.asarray(
        [
            (np.nan, 0.12, 0.20, 0.01),
            (0.60, np.nan, 0.20, 0.01),
            (0.60, 0.12, np.nan, 0.01),
            (0.60, 0.12, 0.20, np.nan),
            (0.60, 0.12, 0.00, 0.01),
            (0.60, 0.20, 0.20, 0.01),
        ],
        dtype=np.float32,
    )
    synthetic_labels = [
        "NaN theta",
        "NaN HH",
        "NaN VV",
        "NaN HV",
        "zero VV",
        "HH/VV = 1",
    ]
    return np.vstack((real_inputs, synthetic_inputs)), labels + synthetic_labels


def _header():
    return (
        f"{'variant':>12} {'iter':>5} {'field':>5} {'exact_diff':>11} "
        f"{'nan_diff':>8} {'finite_n':>9} {'p99_abs':>13} {'max_abs':>13}"
    )


def _summary_row(variant, iterations, field, reference, candidate):
    both_nan = np.isnan(reference) & np.isnan(candidate)
    exact = (reference == candidate) | both_nan
    finite = np.isfinite(reference) & np.isfinite(candidate)
    difference = np.abs(reference[finite] - candidate[finite])
    nan_difference = np.count_nonzero(np.isnan(reference) != np.isnan(candidate))
    p99 = np.percentile(difference, 99) if difference.size else np.nan
    maximum = np.max(difference) if difference.size else np.nan
    return (
        f"{variant:>12} {iterations:5d} {field:>5} "
        f"{np.count_nonzero(~exact):11d} {nan_difference:8d} "
        f"{difference.size:9d} {p99:13.6g} {maximum:13.6g}"
    )


if __name__ == "__main__":
    main()
