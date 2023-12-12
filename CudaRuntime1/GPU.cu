#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>

void saxpy(int n, float a, float* x, int incx, float* y, int incy) {
    for (int i = 0; i < n; i++) {
        if (i * incx >= n || i * incy >= n)
            return;
        y[i * incy] = a * x[i * incx] + y[i * incy];
    }
}

void daxpy(int n, double a, double* x, int incx, double* y, int incy) {
    for (int i = 0; i < n; i++) {
        if (i * incx >= n || i * incy >= n)
            return;
        y[i * incy] = a * x[i * incx] + y[i * incy];
    }
}

void saxpyOMP(int n, float a, const float* x, int incx, float* y, int incy) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (i * incy >= n || i * incx >= n) return;
        y[i * incy] = a * x[i * incx] + y[i * incy];
    }
}

void daxpyOMP(int n, double a, const double* x, int incx, double* y, int incy) {
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (i * incy >= n || i * incx >= n) return;
        y[i * incy] = a * x[i * incx] + y[i * incy];
    }
}

void fill(int n, float* h_x_float, float* h_y_float, double* h_x_double, double* h_y_double) {
    for (int i = 0; i < n; ++i) {
        h_x_float[i] = 1.0f;
        h_y_float[i] = 2.0f;
        h_x_double[i] = 1.0;
        h_y_double[i] = 2.0;
    }
}

__global__ void saxpyGPU(int n, float a, float* x, int incx, float* y, int incy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i * incy >= n || i * incx >= n) return;
        y[i * incy] = a * x[i * incx] + y[i * incy];
    }
}

__global__ void daxpyGPU(int n, double a, double* x, int incx, double* y, int incy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (i * incy >= n || i * incx >= n) return;
        y[i * incy] = a * x[i * incx] + y[i * incy];
    }
}

int main() {
    int n = 180000000;
    float a_float = 2.0f;
    double a_double = 2.0;
    float* x_float, * y_float;
    double* x_double, * y_double;
    const int block_size = 32;
    const int num_block = (n + block_size - 1) / block_size;

    cudaMalloc(&x_float, n * sizeof(float));
    cudaMalloc(&y_float, n * sizeof(float));
    cudaMalloc(&x_double, n * sizeof(double));
    cudaMalloc(&y_double, n * sizeof(double));

    float* h_x_float = new float[n];
    float* h_y_float = new float[n];
    double* h_x_double = new double[n];
    double* h_y_double = new double[n];

    //CPU--------------------------------------------------
    fill(n, h_x_float, h_y_float, h_x_double, h_y_double);
    double start = omp_get_wtime();
    saxpy(n, a_float, h_x_float, 1, h_y_float, 1);
    double finish = omp_get_wtime();
    std::cout <<"CPU saxpy: "<< finish - start << std::endl;

    start = omp_get_wtime();
    daxpy(n, a_double , h_x_double, 1, h_y_double, 1);
    finish = omp_get_wtime();
    std::cout <<"CPU daxpy: "<< finish - start << std::endl;

    float max_error_float = 0.0f;
    double max_error_double = 0.0;
    for (int i = 0; i < n; ++i) {
        max_error_float = std::max(max_error_float, std::abs(h_y_float[i] - 4.0f));
        max_error_double = std::max(max_error_double, std::abs(h_y_double[i] - 4.0));
    }
    std::cout << "Error saxpy: " << max_error_float << std::endl;
    std::cout << "Error daxpy: " << max_error_double << std::endl;
    std::cout << std::endl;

    //GPU--------------------------------------------------
    fill(n, h_x_float, h_y_float, h_x_double, h_y_double);
    cudaMemcpy(x_float, h_x_float, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_float, h_y_float, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x_double, h_x_double, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_double, h_y_double, n * sizeof(double), cudaMemcpyHostToDevice);
    start = omp_get_wtime();
    saxpyGPU << <num_block, block_size >> > (n, a_float, x_float, 1, y_float, 1);
    cudaDeviceSynchronize();
    finish = omp_get_wtime();
    std::cout << "GPU saxpy: " << finish - start << std::endl;

    start = omp_get_wtime();
    daxpyGPU << <num_block, block_size >> > (n, a_double, x_double, 1, y_double, 1);
    cudaDeviceSynchronize();
    finish = omp_get_wtime();
    std::cout<< "GPU daxpy: " << finish - start << std::endl;

    float* checkFloat = new float[n];
    double* checkDouble = new double[n];
    cudaMemcpy(checkFloat, y_float, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(checkDouble, y_double, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        max_error_float = std::max(max_error_float, std::abs(checkFloat[i] - 4.0f));
        max_error_double = std::max(max_error_double, std::abs(checkDouble[i] - 4.0));
    }
    std::cout << "Error saxpy: " << max_error_float << std::endl;
    std::cout << "Error daxpy: " << max_error_double << std::endl;
    std::cout << std::endl;

    cudaFree(x_float);
    cudaFree(y_float);
    cudaFree(x_double);
    cudaFree(y_double);

    //OMP
    fill(n, h_x_float, h_y_float, h_x_double, h_y_double);
    omp_set_num_threads(2);
    start = omp_get_wtime();
    saxpyOMP(n, a_float, h_x_float, 1, h_y_float, 1);
    finish = omp_get_wtime();
    std::cout << "OMP saxpy: " << finish - start << std::endl;

    start = omp_get_wtime();
    daxpyOMP(n, a_double, h_x_double, 1, h_y_double, 1);
    finish = omp_get_wtime();
    std::cout << "OMP daxpy: " << finish - start << std::endl;

    for (int i = 0; i < n; ++i) {
        max_error_float = std::max(max_error_float, std::abs(h_y_float[i] - 4.0f));
        max_error_double = std::max(max_error_double, std::abs(h_y_double[i] - 4.0));
    }
    std::cout << "Error saxpy: " << max_error_float << std::endl;
    std::cout << "Error daxpy: " << max_error_double << std::endl;
    std::cout << std::endl;

    delete[] h_x_float;
    delete[] h_y_float;
    delete[] h_x_double;
    delete[] h_y_double;
    delete[] checkFloat;
    delete[] checkDouble;

    return 1;
}