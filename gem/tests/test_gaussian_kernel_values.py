"""The model-curve kernels must produce the same values after the thread-mapping change.

`gaussian_kernel.cu` was rewritten so consecutive threads cover consecutive pixels of one curve
rather than one thread covering a whole curve -- see test_model_curve_kernel_launch.py for why. The
per-element arithmetic was deliberately left untouched, so this is bit-equality, not closeness.

Verifying that normally needs a GPU, which the development machines do not have. The kernels are
plain C, though, so they can be compiled as host C++ against a handful of stand-ins for the CUDA
builtins and driven with the same grid geometry the launcher produces. That is what
`kernel_host_harness.cpp` does: it runs all four kernels over 60 randomised problem shapes against a
verbatim transcription of the previous kernel bodies and compares the output buffers with memcmp.

It also exercises the `blockIdx.y` grid-stride by running each case a second time with the y cap
lowered to 7, which is the path a chunk holding more than 65535 curves would take and which nothing
on the production configs would otherwise reach.

Skips if no C++ compiler is available.
"""
import os
import shutil
import subprocess

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNELS_DIR = os.path.join(os.path.dirname(_TESTS_DIR), "kernels")
_HARNESS = os.path.join(_TESTS_DIR, "kernel_host_harness.cpp")


def _compiler():
    for candidate in ("c++", "clang++", "g++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def test_kernels_are_bit_identical_to_the_previous_thread_mapping(tmp_path):
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++ compiler available to build the host harness")

    binary = str(tmp_path / "kernel_host_harness")
    build = subprocess.run(
        [compiler, "-O1", "-std=c++17", "-I", _KERNELS_DIR, "-o", binary, _HARNESS],
        capture_output=True, text=True)
    assert build.returncode == 0, f"harness did not compile:\n{build.stderr}"

    run = subprocess.run([binary], capture_output=True, text=True)
    assert run.returncode == 0, f"kernel output changed:\n{run.stdout}\n{run.stderr}"
    assert "bit-identical" in run.stdout, run.stdout
    # guard against the harness silently testing nothing
    cases = int(run.stdout.split()[0])
    assert cases >= 400, f"harness only ran {cases} cases"
