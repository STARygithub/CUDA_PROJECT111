
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

cudaError_t mergeSortCuda(int* table_in, int* table_out, unsigned long int size);

void generateNumbers(int* table1, int* table2, unsigned long int n);
void print_table(int* table, unsigned long int table_size);
int mergeCPU(int* table_in1, int* table_in2, int* table_in, unsigned long int len);
int mergeSortCPU(int* table_in, unsigned long int len);
void myFunction(int* myTable, int size);


__global__ void MaclaurinSeriesKernel(int* array, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = static_cast<float>(array[i]) / 1000.0;  // Adjust x for better convergence
        float result = 1.0;
        float term = 1.0;
        for (int j = 1; j <= 100; j++) {
            term *= x / j;
            result += term;
        }
        array[i] = static_cast<int>(1000.0 * result);
    }
}


void MaclaurinSeries(int* array, int size) {
    for (int i = 0; i < size; i++) {
        float x = static_cast<float>(array[i]) / 1000.0;  // Adjust x for better convergence
        float result = 1.0;
        float term = 1.0;
        for (int j = 1; j <= 100; j++) {
            term *= x / j;
            result += term;
        }
        array[i] = static_cast<int>(1000.0 * result);
    }
}

__global__ void ReverseSortKernel(int* array, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size / 2) {
        int temp = array[i];
        array[i] = array[size - 1 - i];
        array[size - 1 - i] = temp;
    }
}

void ReverseSort(int* array, int size) {
    for (int i = 0; i < size / 2; i++) {
        int temp = array[i];
        array[i] = array[size - 1 - i];
        array[size - 1 - i] = temp;
    }
}



void myFunction(int* myTable, int size) {
    printf("\n");
    printf("len of table = %d\n", size);
    int* buffor = new int[size];

    for (int i = 2048; i <= size; i *= 2) {
        int iterations = size / i;
        for (int j = 0; j < iterations; j++) {
            int start = 0 + j * i;
            int middle = i / 2 + start;
            int startStop = middle;
            int middleStop = i + start;
            for (int k = 0 + i * j; k < i * j + i; k++) {
                if (start < startStop) {
                    if (middle < middleStop) {

                        if (myTable[start] < myTable[middle])
                        {
                            buffor[k] = myTable[start++];
                        }
                        else
                        {
                            buffor[k] = myTable[middle++];
                        }
                    }
                    else
                    {
                        buffor[k] = myTable[start++];

                    }
                }
                else
                {
                    buffor[k] = myTable[middle++];
                }

            }

        }
        for (int l = 0; l < size; l++) { myTable[l] = buffor[l]; }
    }

    delete[] buffor;
}

__global__ void mergeKernelShared2(int* table_in, int* table_out) {
    extern __shared__ int sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int si = threadIdx.x;

    sdata[si] = table_in[i];
    __syncthreads();


    for (unsigned int lvl = 2; lvl <= blockDim.x; lvl *= 2) {
        if (si < blockDim.x / lvl) {
            int idx = threadIdx.x * lvl;
            int start = threadIdx.x * lvl;
            int middle = start + lvl / 2;
            int startStop = middle;
            int middleStop = startStop + lvl / 2;
#pragma unroll
            for (unsigned z = 0; z < lvl; z++) {

                if (start < startStop && middle < middleStop) {
                    if (sdata[start] < sdata[middle]) {
                        sdata[blockDim.x + idx + z] = sdata[start++];
                    }
                    else {
                        sdata[blockDim.x + idx + z] = sdata[middle++];
                    }
                }
                else {
                    if (start < startStop) {
                        sdata[blockDim.x + idx + z] = sdata[start++];
                    }
                    else
                    {
                        sdata[blockDim.x + idx + z] = sdata[middle++];
                    }
                }
            }
            for (unsigned b = threadIdx.x * lvl; b < threadIdx.x * lvl + lvl; b++) {
                sdata[b] = sdata[blockDim.x + b];

            }
        }
        __syncthreads();

    }

    table_out[i] = sdata[si];
}


__global__ void mergeKernel(int* table_in, int* table_out, unsigned long int lvl) {
    unsigned long int i = (blockIdx.x * blockDim.x + threadIdx.x) * lvl;

    unsigned long int start = i;
    unsigned long int middle = i + lvl / 2;
    unsigned long int start1 = i;
    unsigned long int middle1 = i + lvl / 2;
    for (unsigned long int j = i; j < i + lvl; j++) {
        if (start < middle1 && middle < start1 + lvl) {
            if (table_in[start] < table_in[middle]) {
                table_out[j] = table_in[start];
                start++;
            }
            else {
                table_out[j] = table_in[middle];
                middle++;
            }
        }
        else {
            if (start < middle1) {
                table_out[j] = table_in[start];
                start++;
            }
            else {
                table_out[j] = table_in[middle];
                middle++;
            }
        }
    }
}

void generateNumbers(int* table1, int* table2, unsigned long int n)
{
    int random_number = 0;
    for (int i = 0; i < n; i++) {
        random_number = rand() % 1000 + 1;
        table1[i] = random_number;
        table2[i] = random_number;
    }
}

__global__ void dummyKernel() {
    // This kernel does nothing but ensures the GPU is initialized.
}

int mergeCPU(int* table_in1, int* table_in2, int* table_in, unsigned long int len) {
    unsigned long int start = 0;
    unsigned long int start1 = 0;
    unsigned long int middle = 0;
    unsigned long int middle1 = len;
    unsigned long int size = middle1 * 2;
    for (unsigned long int j = 0; j < size; j++) {
        if (start < middle1 && middle < middle1) {
            if (table_in1[start] < table_in2[middle]) {
                table_in[j] = table_in1[start];
                start++;
            }
            else {
                table_in[j] = table_in2[middle];
                middle++;
            }
        }
        else {
            if (start < middle1) {
                table_in[j] = table_in1[start];
                start++;
            }
            else {
                table_in[j] = table_in2[middle];
                middle++;
            }

        }

    }

    return *table_in;
}


int mergeSortCPU(int* table_in, unsigned long int len) {

    unsigned long int table_length = len;
    unsigned long int half_length = table_length / 2;
    if (table_length <= 1) {
        return *table_in;
    }

    int* first_half = new int[half_length];
    int* second_half = new int[half_length];
    for (int i = 0; i < half_length; i++) {
        first_half[i] = table_in[i];
        second_half[i] = table_in[i + half_length];
    }

    *first_half = mergeSortCPU(first_half, half_length);
    *second_half = mergeSortCPU(second_half, half_length);
    //print_table(first_half, half_length);
    //print_table(second_half, half_length);
    *table_in = mergeCPU(first_half, second_half, table_in, half_length);
    //print_table(table_in, table_length);
    delete[] first_half;
    delete[] second_half;

    return *table_in;
}

void print_table(int* table, unsigned long int table_size) {
    printf("table = { ");
    for (unsigned long int i = 0; i < table_size - 1; i++)
        printf("%d, ", table[i]);
    printf("%d}\n", table[table_size - 1]);
}

int main() {
    dummyKernel << <1, 1 >> > ();
    cudaDeviceSynchronize();
    const int MAX_THREADS = 1024;

    unsigned long int dynamic_size = 33554432;

    int* dynamic_table = new int[dynamic_size];
    int* dynamic_table_CPU = new int[dynamic_size];
    int* dynamic_table_out = new int[dynamic_size];

    generateNumbers(dynamic_table, dynamic_table_CPU, dynamic_size);

    auto start_CPU = chrono::high_resolution_clock::now();
    mergeSortCPU(dynamic_table_CPU, dynamic_size);
    auto end_CPU = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_CPU = end_CPU - start_CPU;

    std::cout << "CPU Sort Time: " << duration_CPU.count() << " seconds\n";

    auto start_GPU = chrono::high_resolution_clock::now();
    cudaError_t cudaStatus1 = mergeSortCuda(dynamic_table, dynamic_table_out, dynamic_size);
    auto end_GPU = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_GPU = end_GPU - start_GPU;

    std::cout << "GPU Sort Time: " << duration_GPU.count() << " seconds\n";

    

   

    



    // Display a few elements from the arrays
    int step = 1;
    printf("CPU: ");
    for (int i = 0; i < 16; i += 1) {
        printf("%d ", dynamic_table_CPU[i * step]);
    }
    printf("%d %d", dynamic_table_CPU[dynamic_size / 2], dynamic_table_CPU[dynamic_size - 1]);
    printf("\nGPU: ");
    for (int i = 0; i < 16; i += 1) {
        printf("%d ", dynamic_table_out[i * step]);
    }
    printf("%d %d", dynamic_table_out[dynamic_size / 2], dynamic_table_out[dynamic_size - 1]);



    auto start_CPU_math = chrono::high_resolution_clock::now();
    MaclaurinSeries(dynamic_table_CPU, dynamic_size);
    auto end_CPU_math = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_CPU_math = end_CPU_math - start_CPU_math;

    std::cout << "\nCPU MaclaurinSeries Operation Time: " << duration_CPU_math.count() << " seconds\n";

    
    auto start_GPU_math = chrono::high_resolution_clock::now();
    MaclaurinSeriesKernel << <(dynamic_size + 1023) / 1024, 1024 >> > (dynamic_table_out, dynamic_size);
    cudaDeviceSynchronize();
    auto end_GPU_math = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_GPU_math = end_GPU_math - start_GPU_math;

    std::cout << "GPU MaclaurinSeries Operation Time: " << duration_GPU_math.count() << " seconds\n";

   


  
    auto start_CPU_reverse = chrono::high_resolution_clock::now();
    ReverseSort(dynamic_table_CPU, dynamic_size);
    auto end_CPU_reverse = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_CPU_reverse = end_CPU_reverse - start_CPU_reverse;

    std::cout << "CPU ReverseSort Time: " << duration_CPU_reverse.count() << " seconds\n";




    auto start_GPU_reverse = chrono::high_resolution_clock::now();
    ReverseSortKernel << <(dynamic_size + 1023) / 1024, 1024 >> > (dynamic_table_out, dynamic_size);
    cudaDeviceSynchronize();
    auto end_GPU_reverse = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_GPU_reverse = end_GPU_reverse - start_GPU_reverse;

    std::cout << "GPU ReverseSort Time: " << duration_GPU_reverse.count() << " seconds\n";

    



    // Clean up
    delete[] dynamic_table;
    delete[] dynamic_table_CPU;
    delete[] dynamic_table_out;

    cudaStatus1 = cudaDeviceReset();
    if (cudaStatus1 != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t mergeSortCuda(int* table_in, int* table_out, unsigned long int size) {

    int* device_table_in = 0;
    int* device_table_in2 = 0;
    int* device_table_out = 0;
    int* device_table_out2 = 0;
    unsigned long int number_of_blocks = 1;
    unsigned long int number_of_threads = size;

    unsigned long int halfSize = size / 2;
    size_t halfSizeInBytes = halfSize * sizeof(int);
    int* secondHalfDevicePtr = device_table_in + halfSize;
    int* secondHalfHostPtr = device_table_out + halfSize;


    cudaError_t cudaStatus;
    cudaStream_t stream1, stream2;

    cudaError_t r1 = cudaStreamCreate(&stream1);
    cudaError_t r2 = cudaStreamCreate(&stream2);
    auto start_k = chrono::high_resolution_clock::now();

    //now choose device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&device_table_out, size / 2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 11");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&device_table_in, size / 2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 21");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&device_table_out2, size / 2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 12");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&device_table_in2, size / 2 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 22");
        goto Error;
    }
    auto end_k = chrono::high_resolution_clock::now();
    chrono::duration<float> duration_k = end_k - start_k;
    std::cout << " | data alocation Time = " << duration_k.count() << endl;


    cudaStatus = cudaMemcpyAsync(device_table_in, table_in, (size / 2) * sizeof(int), cudaMemcpyHostToDevice, stream1);
    mergeKernelShared2 << < (size / 1024) / 2, 1024, 1024 * 2 * 4, stream1 >> > (device_table_in, device_table_out);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "1 kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    cudaStatus = cudaMemcpyAsync(device_table_in2, table_in + halfSize, (size / 2) * sizeof(int), cudaMemcpyHostToDevice, stream2);
    mergeKernelShared2 << < (size / 1024) / 2, 1024, 1024 * 2 * 4, stream2 >> > (device_table_in2, device_table_out2);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "2 kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }

    /*
    cudaStatus = cudaMemcpy(table_out, device_table_out, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    */

    cudaStatus = cudaMemcpyAsync(table_out, device_table_out, (size / 2) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpyAsync(table_out + halfSize, device_table_out2, (size / 2) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    table_in = table_out;
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

Error:
    cudaFree(device_table_in);
    cudaFree(device_table_out);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    return cudaStatus;
}
