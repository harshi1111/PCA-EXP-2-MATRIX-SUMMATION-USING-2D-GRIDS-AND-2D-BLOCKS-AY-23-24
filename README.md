# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>NAME : HARSHITHA V</h3>
<h3>REGISTER NO : 212223230074</h3>
<h3>EX. NO : 2</h3>
<h3>DATE : 20-10-2024</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```
%%writefile sumMatrixOnGPU-2D-grid-2D-block.cu
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

#define N 1024 // Size of the matrix

__global__ void sumMatrixOnGPU2D(int *A, int *B, int *C, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < width) {
        C[y * width + x] = A[y * width + x] + B[y * width + x];
    }
}

void initialData(int *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = rand() % 100; // Random values between 0 and 99
    }
}

void sumMatrixOnHost(int *A, int *B, int *C, int width) {
    for (int i = 0; i < width * width; i++) {
        C[i] = A[i] + B[i];
    }
}

void checkResult(int *C, int *C_ref, int size) {
    for (int i = 0; i < size; i++) {
        if (C[i] != C_ref[i]) {
            printf("Mismatch at index %d: GPU %d, Host %d\n", i, C[i], C_ref[i]);
            return; // Stop at the first mismatch
        }
    }
    printf("All results are correct!\n");
}

int main() {
    int size = N * N * sizeof(int);
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);
    int *h_C_ref = (int *)malloc(size);
    
    // Initialize matrices
    srand(time(0));
    initialData(h_A, N * N);
    initialData(h_B, N * N);

    // Timing for host
    clock_t startHost = clock();
    sumMatrixOnHost(h_A, h_B, h_C_ref, N);
    clock_t endHost = clock();
    double hostTime = double(endHost - startHost) / CLOCKS_PER_SEC;

    int *d_A, *d_B, *d_C;

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Timing for device
    cudaEvent_t startDevice, stopDevice;
    cudaEventCreate(&startDevice);
    cudaEventCreate(&stopDevice);
    cudaEventRecord(startDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16); // Each block will have 16x16 threads
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    sumMatrixOnGPU2D<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stopDevice);
    cudaEventSynchronize(stopDevice);

    float deviceTime;
    cudaEventElapsedTime(&deviceTime, startDevice, stopDevice);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Check the result
    checkResult(h_C, h_C_ref, N * N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    cudaDeviceReset();

    // Print execution times
    printf("Host computation time: %.6f seconds\n", hostTime);
    printf("Device computation time: %.6f seconds\n", deviceTime / 1000); // Convert ms to seconds

    return 0;
}

```

## OUTPUT:
![image](https://github.com/user-attachments/assets/0adcb9d6-8028-4e1d-9217-9b3b3d221ceb)



## RESULT:
The host took 0.004803 seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in 0.102751 seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
