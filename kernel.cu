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

// Global Memory Kernel für Matrixmultiplikation
__global__ void matrixMulGlobal(float* a, float* b, float* c, int width)
{
    // berechnen der globalen Zeilen- und Spaltenindizes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // nur berechnen, falls innerhalb der Matrixgrenzen
    if (row < width && col < width)
    {
        float sum = 0.0f;

        // Skalarrodukt
        for (int k = 0; k < width; k++)
        {
            sum += a[row * width + k] * b[k * width + col];
        }

        c[row * width + col] = sum;
    }
}

// Wrapper-Funktion, die das CUDA-Speichermanagement übernimmt
void matMulGPUGlobal(float* matrixA, float* matrixB, float* matrixC, int dimSize)
{
    int size = dimSize * dimSize * sizeof(float);
    float *d_A, *d_B, *d_C;

    gpuErrCheck(cudaMalloc(&d_A, size));
    gpuErrCheck(cudaMalloc(&d_B, size));
    gpuErrCheck(cudaMalloc(&d_C, size));

    gpuErrCheck(cudaMemcpy(d_A, matrixA, size, cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(d_B, matrixB, size, cudaMemcpyHostToDevice));

    // 32x32 threads pro Block ist Standard
    dim3 blockDim(32, 32);
    dim3 gridDim((dimSize + blockDim.x - 1) / blockDim.x,
                 (dimSize + blockDim.y - 1) / blockDim.y);

    matrixMulGlobal<<<gridDim, blockDim>>>(d_A, d_B, d_C, dimSize);
    gpuErrCheck(cudaGetLastError());

    gpuErrCheck(cudaMemcpy(matrixC, d_C, size, cudaMemcpyDeviceToHost));

    gpuErrCheck(cudaFree(d_A));
    gpuErrCheck(cudaFree(d_B));
    gpuErrCheck(cudaFree(d_C));
}

#define TILE_SIZE 32

// Shared Memory Kernel für Matrixmultiplikation
__global__ void matrixMulShared(float* a, float* b, float* c, int width)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // durchlaufe alle Tiles
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // lade Daten in den Shared Memory
        if (row < width && t * TILE_SIZE + threadIdx.x < width)
        {
            // lade ein Element von Matrix A
            tileA[threadIdx.y][threadIdx.x] = a[row * width + t * TILE_SIZE + threadIdx.x];
        } 
        else
        {
            // ausserhalb der Matrix, setze auf 0
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (t * TILE_SIZE + threadIdx.y < width && col < width)
        {
            // lade ein Element von Matrix B
            tileB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * width + col];
        }
        else
        {
            // ausserhalb der Matrix, setze auf 0
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // syhchronisiere, um sicherzustellen, dass alle Daten geladen sind
        __syncthreads();

        // berechne Teilsumme für diesen Tile
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // schreibe das Ergebnis, falls innerhalb der Matrixgrenzen
    if (row < width && col < width)
    {
        c[row * width + col] = sum;
    }
}

// Wrapper-Funktion für Shared Memory
void matMulGPUShared(float* matrixA, float* matrixB, float* matrixC, int dimSize)
{
    int size = dimSize * dimSize * sizeof(float);
    float *d_A, *d_B, *d_C;

    gpuErrCheck(cudaMalloc(&d_A, size));
    gpuErrCheck(cudaMalloc(&d_B, size));
    gpuErrCheck(cudaMalloc(&d_C, size));

    gpuErrCheck(cudaMemcpy(d_A, matrixA, size, cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(d_B, matrixB, size, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimSize + blockDim.x - 1) / blockDim.x,
                 (dimSize + blockDim.y - 1) / blockDim.y);

    matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, dimSize);
    gpuErrCheck(cudaGetLastError());

    gpuErrCheck(cudaMemcpy(matrixC, d_C, size, cudaMemcpyDeviceToHost));

    gpuErrCheck(cudaFree(d_A));
    gpuErrCheck(cudaFree(d_B));
    gpuErrCheck(cudaFree(d_C));
}

// Hilfsfunktion zum vergleichen von Ergebnissen
bool checkResults(float* reference, float* result, int size)
{
    float epsilon = 1e-5; // Toleranz für Rundungsfehler
    for (int i = 0; i < size; i++)
    {
        if (abs(reference[i] - result[i]) > epsilon)
        {
            cout << "Validation failed at index " << i
                 << ": " << reference[i] << " vs " << result[i] << endl;
            return false;
        }
    }
    cout << "Validation successful!" << endl;
    return true;
}

int main()
{
    // ATTENTION: Your code must be robust in regards of this number.
    // ATTENTION: DIM_SIZE of 4096 is maybe not a good idea during development :)
    // DIM_SIZE can and will change during the assessment, also to non 2^n values!
    for (int DIM_SIZE = 64; DIM_SIZE <= 2048; DIM_SIZE *= 2) {
        cout << "DIM_SIZE: " << DIM_SIZE << endl;

        float* h_matrixA = new float[DIM_SIZE * DIM_SIZE];
        float* h_matrixB = new float[DIM_SIZE * DIM_SIZE];
        float* h_matrixC_cpu = new float[DIM_SIZE * DIM_SIZE];
        float* h_matrixC_global = new float[DIM_SIZE * DIM_SIZE];
        float* h_matrixC_shared = new float[DIM_SIZE * DIM_SIZE];

        populateMatrixBuffer(h_matrixA, DIM_SIZE);
        populateMatrixBuffer(h_matrixB, DIM_SIZE);

        auto startTime = chrono::high_resolution_clock::now();
        matMulCPUNaive(h_matrixA, h_matrixB, h_matrixC_cpu, DIM_SIZE);
        auto endTime = chrono::high_resolution_clock::now();
        auto cpuTime = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
        cout << "CPU time [ms]: " << cpuTime << endl;

        /* GPU Events für Zeitmessung */
        cudaEvent_t start, stop;
        gpuErrCheck(cudaEventCreate(&start));
        gpuErrCheck(cudaEventCreate(&stop));

        /* GPU Global Memory Implementierung */
        // Warm-up (für stabilere Messung)
        matMulGPUGlobal(h_matrixA, h_matrixB, h_matrixC_global, DIM_SIZE);

        // Tatsächliche Messung
        gpuErrCheck(cudaEventRecord(start));
        matMulGPUGlobal(h_matrixA, h_matrixB, h_matrixC_global, DIM_SIZE);
        gpuErrCheck(cudaEventRecord(stop));
        gpuErrCheck(cudaEventSynchronize(stop));

        float gpuGlobalTime = 0.0f;
        gpuErrCheck(cudaEventElapsedTime(&gpuGlobalTime, start, stop));
        cout << "GPU Global Memory time [ms]: " << gpuGlobalTime << endl;

        // Validiere Global Memory Ergebnisse
        if (!checkResults(h_matrixC_cpu, h_matrixC_global, DIM_SIZE * DIM_SIZE))
        {
            cout << "ERROR: Results don't match!" << endl;
        }

        /* GPU Shared Memory Implementierung */
        // Warm-up (für stabilere Messung)
        matMulGPUShared(h_matrixA, h_matrixB, h_matrixC_shared, DIM_SIZE);

        gpuErrCheck(cudaEventRecord(start));
        matMulGPUShared(h_matrixA, h_matrixB, h_matrixC_shared, DIM_SIZE);
        gpuErrCheck(cudaEventRecord(stop));
        gpuErrCheck(cudaEventSynchronize(stop));

        float gpuSharedTime = 0.0f;
        gpuErrCheck(cudaEventElapsedTime(&gpuSharedTime, start, stop));
        cout << "GPU Shared Memory time [ms]: " << gpuSharedTime << endl;

        // validiere Shared Memory Ergebnisse
        if (!checkResults(h_matrixC_cpu, h_matrixC_shared, DIM_SIZE * DIM_SIZE))
        {
            cout << "ERROR: Shared Memory results don't match!" << endl;
        }

        // berechne und zeige die Speedups
        float globalSpeedup = cpuTime / gpuGlobalTime;
        float sharedSpeedup = cpuTime / gpuSharedTime;
        float sharedVsGlobalSpeedup = gpuGlobalTime / gpuSharedTime;

        cout << "Global Memory Speedup vs CPU: " << globalSpeedup << "x" << endl;
        cout << "Shared Memory Speedup vs CPU: " << sharedSpeedup << "x" << endl;
        cout << "Shared Memory Speedup vs Global Memory: " << sharedVsGlobalSpeedup << "x" << endl;

        gpuErrCheck(cudaEventDestroy(start));
        gpuErrCheck(cudaEventDestroy(stop));

        delete[] h_matrixA;
        delete[] h_matrixB;
        delete[] h_matrixC_cpu;
        delete[] h_matrixC_global;
        delete[] h_matrixC_shared;

        cout << "-------------------------------------" << endl;
    }

    return 0;
}