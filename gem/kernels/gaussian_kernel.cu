extern "C"
{
	// ---------------------------------------------------------------------------------------------
	// Thread mapping
	//
	// All four kernels are launched with consecutive threads over consecutive PIXELS of one model
	// curve, and one block-row (blockIdx.y) per pRF point:
	//
	//     block  (256, 1, 1)
	//     grid   (ceil(num_pixels / 256), min(num_curves, 65535), 1)
	//
	// They used to run one thread per pRF point, with that thread walking its whole curve. Two
	// things were wrong with that. A 15,000-curve chunk launched ceil(15000/512) = 30 blocks, which
	// is fewer blocks than a V100 has SMs, so five eighths of the device sat idle. And lanes within
	// a warp wrote addresses num_pixels doubles apart, so every 8-byte store was its own 32-byte
	// memory transaction.
	//
	// Each output element is still computed by the same expression from the same operands, so the
	// values produced are unchanged.
	//
	// The loop over blockIdx.y is a grid-stride because gridDim.y is capped at the CUDA limit of
	// 65535; a chunk holding more curves than that is still covered, one stride at a time.
	// ---------------------------------------------------------------------------------------------

	// Send (ux, uy and sigma) in a single flattened array
	__global__ void gc_using_args_arrays_cuda_Kernel(
		double* result_gaussian_curves,
		double* prfPointsArgsFlatArr,
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int num_dimensions,	// for a Gaussian model, num_dimensions = 3
		int nStimulusRows,
		int nStimulusCols,
		int numTotalGaussianCurves
	)
	{
		const int numPixels = nStimulusRows * nStimulusCols;
		const int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (pixIdx >= numPixels) return;

		// The stimulus point this thread is responsible for is the same for every curve, so it is
		// read once here rather than once per curve.
		const int stim_vf_row = pixIdx / nStimulusCols;
		const int stim_vf_col = pixIdx - (stim_vf_row * nStimulusCols);
		const double y = stimulus_vf_points_y[stim_vf_row];
		const double x = stimulus_vf_points_x[stim_vf_col];

		for (int prfPointIdx = blockIdx.y; prfPointIdx < numTotalGaussianCurves; prfPointIdx += gridDim.y)
		{
			const double x_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions];
			const double y_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 1];
			const double sigma = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 2];

			// size_t so the flat index cannot overflow: 15,000 curves x 90,601 pixels is already
			// 1.36e9, within 1.6x of what a signed 32-bit index can hold.
			const size_t gaussIdx = (size_t)prfPointIdx * (size_t)numPixels + (size_t)pixIdx;

			const double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);

			result_gaussian_curves[gaussIdx] = exp(exponent);
		}
	}

	// Send (ux, uy and sigma) directly as arrays
	__global__ void dgc_dx_using_args_arrays_cuda_Kernel(
		double* result_Dx_gaussian_curves,
		double* prfPointsArgsFlatArr,
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int num_dimensions,	// for a Gaussian model, num_dimensions = 3
		int nStimulusRows,
		int nStimulusCols,
		int numTotalGaussianCurves
	)
	{
		const int numPixels = nStimulusRows * nStimulusCols;
		const int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (pixIdx >= numPixels) return;

		const int stim_vf_row = pixIdx / nStimulusCols;
		const int stim_vf_col = pixIdx - (stim_vf_row * nStimulusCols);
		const double y = stimulus_vf_points_y[stim_vf_row];
		const double x = stimulus_vf_points_x[stim_vf_col];

		for (int prfPointIdx = blockIdx.y; prfPointIdx < numTotalGaussianCurves; prfPointIdx += gridDim.y)
		{
			const double x_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions];
			const double y_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 1];
			const double sigma = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 2];

			const size_t gaussIdx = (size_t)prfPointIdx * (size_t)numPixels + (size_t)pixIdx;

			const double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);

			result_Dx_gaussian_curves[gaussIdx] = ((x - x_mean) / (sigma * sigma)) * exp(exponent);
		}
	}

	__global__ void dgc_dy_using_args_arrays_cuda_Kernel(
		double* result_Dy_gaussian_curves,
		double* prfPointsArgsFlatArr,
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int num_dimensions,	// for a Gaussian model, num_dimensions = 3
		int nStimulusRows,
		int nStimulusCols,
		int numTotalGaussianCurves
	)
	{
		const int numPixels = nStimulusRows * nStimulusCols;
		const int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (pixIdx >= numPixels) return;

		const int stim_vf_row = pixIdx / nStimulusCols;
		const int stim_vf_col = pixIdx - (stim_vf_row * nStimulusCols);
		const double y = stimulus_vf_points_y[stim_vf_row];
		const double x = stimulus_vf_points_x[stim_vf_col];

		for (int prfPointIdx = blockIdx.y; prfPointIdx < numTotalGaussianCurves; prfPointIdx += gridDim.y)
		{
			const double x_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions];
			const double y_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 1];
			const double sigma = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 2];

			const size_t gaussIdx = (size_t)prfPointIdx * (size_t)numPixels + (size_t)pixIdx;

			const double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);

			result_Dy_gaussian_curves[gaussIdx] = ((y - y_mean) / (sigma * sigma)) * exp(exponent);
		}
	}

	__global__ void dgc_dsigma_using_args_arrays_cuda_Kernel(
		double* result_Dsigma_gaussian_curves,
		double* prfPointsArgsFlatArr,
		double* stimulus_vf_points_x,
		double* stimulus_vf_points_y,
		int num_dimensions,	// for a Gaussian model, num_dimensions = 3
		int nStimulusRows,
		int nStimulusCols,
		int numTotalGaussianCurves
	)
	{
		const int numPixels = nStimulusRows * nStimulusCols;
		const int pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (pixIdx >= numPixels) return;

		const int stim_vf_row = pixIdx / nStimulusCols;
		const int stim_vf_col = pixIdx - (stim_vf_row * nStimulusCols);
		const double y = stimulus_vf_points_y[stim_vf_row];
		const double x = stimulus_vf_points_x[stim_vf_col];

		for (int prfPointIdx = blockIdx.y; prfPointIdx < numTotalGaussianCurves; prfPointIdx += gridDim.y)
		{
			const double x_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions];
			const double y_mean = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 1];
			const double sigma = prfPointsArgsFlatArr[prfPointIdx*num_dimensions + 2];

			const size_t gaussIdx = (size_t)prfPointIdx * (size_t)numPixels + (size_t)pixIdx;

			const double exponent = -((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (2 * sigma * sigma);

			result_Dsigma_gaussian_curves[gaussIdx] = (((x - x_mean) * (x - x_mean) + (y - y_mean) * (y - y_mean)) / (sigma * sigma * sigma)) * exp(exponent); //NOTE: removed "minus" sign
		}
	}
}
