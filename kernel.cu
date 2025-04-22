#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>
#include <chrono>

using namespace std;


/* This is our CUDA call wrapper, we will use in PAC.
*
*  Almost all CUDA calls should be wrapped with this makro.
*  Errors from these calls will be catched and printed on the console.
*  If an error appears, the program will terminate.
*
* Example: gpuErrCheck(cudaMalloc(&deviceA, N * sizeof(int)));
*          gpuErrCheck(cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice));
*/
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}


/* Helper which populates a matrix buffer (dimSize*dimSize).
* 
* Think of this as it would load the data from disk or somewhere else.
* This dummy data is only used to fill the buffer as fast as possible.
*/
void populateMatrixBuffer(float* buffer, int dimSize)
{
    // Init of matrix buffer
    for (int i = 0; i < dimSize; i++) {
        for (int j = 1; j <= dimSize; j++) {
            buffer[i * dimSize + j] = 1.0f / j;
        }
    }
}


// Compare result arrays CPU vs GPU result. If no diff, the result pass.
int compareResultVec(float* matrixCPU, float* matrixGPU, int size)
{
    float error = 0;
    for (int i = 0; i < size; i++)
    {
        error += abs(matrixCPU[i] - matrixGPU[i]);
    }
    if (error == 0)  // Is this sane? Think about float processing!
    {
        cout << "Test passed." << endl;
        return 0;
    }
    else
    {
        cout << "Accumulated error: " << error << endl;
        return -1;
    }
}


/* Slow MatMul on the CPU, stores matrixA * matrixB in buffer matrixC
* 
* This is our CPU baseline.
*/
void matMulCPUNaive(float* matrixA, float* matrixB, float* matrixC, int dimSize)
{
    float sum;
    for (int i = 0; i < dimSize; i++)
    {
        for (int j = 0; j < dimSize; j++)
        {
            sum = 0.0;
            for (int n = 0; n < dimSize; n++)
            {
                sum += matrixA[i * dimSize + n] * matrixB[n * dimSize + j];
            }
            matrixC[i * dimSize + j] = sum;
        }
    }
}


int main()
{
    // ATTENTION: Your code must be robust in regards of this number.
    // ATTENTION: DIM_SIZE of 4096 is maybe not a good idea during development :)
    // DIM_SIZE can and will change during the assessment, also to non 2^n values!
    for (int DIM_SIZE = 64; DIM_SIZE <= 4096; DIM_SIZE <<= 1) {
        cout << "DIM_SIZE: " << DIM_SIZE << endl;
        float* h_matrixA = new float[DIM_SIZE * DIM_SIZE];
        float* h_matrixB = new float[DIM_SIZE * DIM_SIZE];
        float* h_matrixC = new float[DIM_SIZE * DIM_SIZE];
        populateMatrixBuffer(h_matrixA, DIM_SIZE);
        populateMatrixBuffer(h_matrixB, DIM_SIZE);

        auto startTime = chrono::high_resolution_clock::now();
        matMulCPUNaive(h_matrixA, h_matrixB, h_matrixC, DIM_SIZE);
        auto endTime = chrono::high_resolution_clock::now();
        cout << "CPU time [ms]: " << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() << endl;

        delete[] h_matrixA;
        delete[] h_matrixB;
        delete[] h_matrixC;
    }

    return 0;
}