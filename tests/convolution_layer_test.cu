// Include necessary headers for CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void convolutionKernel(const float* input, const float* filter, float* output,
                                  int inputHeight, int inputWidth, int inputDepth,
                                  int filterHeight, int filterWidth, int numFilters,
                                  int strideH, int strideW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the position in the output
    int outDepth = idx % numFilters;
    idx /= numFilters;
    int outCol = idx % ((inputWidth - filterWidth) / strideW + 1);
    idx /= ((inputWidth - filterWidth) / strideW + 1);
    int outRow = idx;

    // Compute the starting position in the input for this output position
    int inRowStart = outRow * strideH;
    int inColStart = outCol * strideW;

    // Perform the convolution
    float sum = 0.0f;
    for (int fRow = 0; fRow < filterHeight; ++fRow) {
        for (int fCol = 0; fCol < filterWidth; ++fCol) {
            for (int inDepth = 0; inDepth < inputDepth; ++inDepth) {
                int inRow = inRowStart + fRow;
                int inCol = inColStart + fCol;
                int inIdx = (inRow * inputWidth + inCol) * inputDepth + inDepth;
                int filterIdx = ((fRow * filterWidth) + fCol) * inputDepth + inDepth;
                sum += input[inIdx] * filter[filterIdx + outDepth * filterHeight * filterWidth];
            }
        }
    }

    // Store the result
    int outIdx = (outRow * ((inputWidth - filterWidth) / strideW + 1) + outCol) * numFilters + outDepth;
    output[outIdx] = sum;
}

void ConvolutionLayer::Forward(const arma::cube& input, arma::cube& output) {
    // Allocate memory on the GPU
    float* d_input;
    float* d_output;
    float* d_filters;

    cudaMalloc((void**)&d_input, input.n_elem * sizeof(float));
    cudaMalloc((void**)&d_output, output.n_elem * sizeof(float));
    cudaMalloc((void**)&d_filters, filters_.size() * sizeof(float));

    // Copy input and filters from host to device
    cudaMemcpy(d_input, input.memptr(), input.n_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filters, filters_.memptr(), filters_.n_elem * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int numThreads = 256;
    int numBlocks = (output.n_elem + numThreads - 1) / numThreads;

    // Launch the kernel
    convolutionKernel<<<numBlocks, numThreads>>>(d_input, d_filters, d_output,
                                                  input.n_rows, input.n_cols, input.n_slices,
                                                  filterHeight_, filterWidth_, numFilters_,
                                                  strideH_, strideW_);

    // Copy the result back to host
    cudaMemcpy(output.memptr(), d_output, output.n_elem * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filters);
}

