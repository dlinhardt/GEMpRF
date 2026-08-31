// Compile gaussian_kernel.cu as host C++ and check that the pixel-major thread mapping produces
// byte-identical output to the previous one-thread-per-curve mapping, for all four kernels.
//
// The kernels' arithmetic was deliberately left untouched when the thread mapping changed, so this
// is an equality check, not a closeness check. Driven by test_gaussian_kernel_values.py, which
// compiles and runs it; no GPU or CUDA toolkit is involved.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

// Minimal stand-ins for the CUDA builtins the kernels read.
struct Dim3 { int x, y, z; };
static Dim3 blockIdx, threadIdx, blockDim, gridDim;
#define __global__

#include "gaussian_kernel.cu"

// ---------------------------------------------------------------------------------------------
// The previous kernel bodies, transcribed verbatim: one thread per pRF point, walking its whole
// curve with a running `gaussIdx`.
// ---------------------------------------------------------------------------------------------
enum Term { TERM_VALUE, TERM_DX, TERM_DY, TERM_DSIGMA };

static void old_kernel(Term term, double* out, const double* pts, const double* xr, const double* yr,
                       int nd, int nRows, int nCols, int nCurves) {
    for (int prfPointIdx = 0; prfPointIdx < nCurves; prfPointIdx++) {
        double x_mean = pts[prfPointIdx*nd];
        double y_mean = pts[prfPointIdx*nd + 1];
        double sigma  = pts[prfPointIdx*nd + 2];
        int gaussIdx = prfPointIdx * (nCols * nRows);
        for (int stim_vf_row = 0; stim_vf_row < nRows; stim_vf_row++) {
            for (int stim_vf_col = 0; stim_vf_col < nCols; stim_vf_col++) {
                double y = yr[stim_vf_row];
                double x = xr[stim_vf_col];
                double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);
                switch (term) {
                    case TERM_VALUE:
                        out[gaussIdx] = exp(exponent); break;
                    case TERM_DX:
                        out[gaussIdx] = ((x - x_mean) / (sigma * sigma)) * exp(exponent); break;
                    case TERM_DY:
                        out[gaussIdx] = ((y - y_mean) / (sigma * sigma)) * exp(exponent); break;
                    case TERM_DSIGMA:
                        out[gaussIdx] = (((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (sigma * sigma * sigma)) * exp(exponent); break;
                }
                gaussIdx++;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Drive the current kernels with the geometry __set_kernel_config() produces.
// ---------------------------------------------------------------------------------------------
typedef void (*KernelFn)(double*, double*, double*, double*, int, int, int, int);

static void launch(KernelFn kernel, int nCurves, int nPixels, int capY, double* out, double* pts,
                   double* xr, double* yr, int nd, int nRows, int nCols) {
    blockDim = {256, 1, 1};
    int bx = (nPixels + blockDim.x - 1) / blockDim.x;
    int by = nCurves < capY ? nCurves : capY;
    if (bx < 1) bx = 1;
    if (by < 1) by = 1;
    gridDim = {bx, by, 1};
    for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++)
        for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++)
            for (threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x++)
                kernel(out, pts, xr, yr, nd, nRows, nCols, nCurves);
}

int main() {
    srand(7);
    const KernelFn kernels[4] = {
        gc_using_args_arrays_cuda_Kernel,
        dgc_dx_using_args_arrays_cuda_Kernel,
        dgc_dy_using_args_arrays_cuda_Kernel,
        dgc_dsigma_using_args_arrays_cuda_Kernel,
    };
    const Term terms[4] = { TERM_VALUE, TERM_DX, TERM_DY, TERM_DSIGMA };
    const char* names[4] = { "value", "d/dx", "d/dy", "d/dsigma" };
    // capY 65535 is the production path; capY 7 forces the blockIdx.y grid-stride to wrap.
    const int caps[2] = { 65535, 7 };

    int cases = 0, fails = 0;
    for (int trial = 0; trial < 60; trial++) {
        int nd = 3;
        int nRows   = 1 + rand() % 23;
        int nCols   = 1 + rand() % 23;
        int nCurves = 1 + rand() % 40;
        int nPix = nRows * nCols;

        std::vector<double> pts(nCurves * nd), xr(nCols), yr(nRows);
        for (auto& v : pts) v = (rand() / (double)RAND_MAX) * 6 - 3;
        for (int i = 0; i < nCurves; i++)
            pts[i*nd + 2] = 0.2 + (rand() / (double)RAND_MAX) * 3;   // sigma strictly > 0
        for (auto& v : xr) v = (rand() / (double)RAND_MAX) * 10 - 5;
        for (auto& v : yr) v = (rand() / (double)RAND_MAX) * 10 - 5;

        for (int k = 0; k < 4; k++) {
            for (int c = 0; c < 2; c++) {
                std::vector<double> want(nCurves * nPix, -1.0), got(nCurves * nPix, -2.0);
                old_kernel(terms[k], want.data(), pts.data(), xr.data(), yr.data(), nd, nRows, nCols, nCurves);
                launch(kernels[k], nCurves, nPix, caps[c], got.data(), pts.data(), xr.data(), yr.data(), nd, nRows, nCols);
                cases++;
                if (memcmp(want.data(), got.data(), want.size() * sizeof(double)) != 0) {
                    printf("MISMATCH %s: rows=%d cols=%d curves=%d capY=%d\n",
                           names[k], nRows, nCols, nCurves, caps[c]);
                    fails++;
                }
            }
        }
    }
    printf("%d cases, %d mismatches -> %s\n", cases, fails, fails ? "FAIL" : "bit-identical");
    return fails != 0;
}
