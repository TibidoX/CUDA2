//#include <iostream>
//#include <omp.h>
//#include <cuda_runtime.h>
//#include "GPU.h"
//void saxpy(int n, float a, float* x, int incx, float* y, int incy) {
//    for (int i = 0; i < n; i++) {
//        if (i * incx >= n || i * incy >= n)
//            return;
//        y[i * incy] = a * x[i * incx] + y[i * incy];
//    }
//}
//
//void daxpy(int n, double a, double* x, int incx, double* y, int incy) {
//    for (int i = 0; i < n; i++) {
//        if (i * incx >= n || i * incy >= n)
//            return;
//        y[i * incy] = a * x[i * incx] + y[i * incy];
//    }
//}
//
//void fill(int n, float* h_x_float, float* h_y_float, double* h_x_double, double* h_y_double) {
//    for (int i = 0; i < n; ++i) {
//        h_x_float[i] = 1.0f;
//        h_y_float[i] = 2.0f;
//        h_x_double[i] = 1.0;
//        h_y_double[i] = 2.0;
//    }
//}
//
////int main() {
////    int n = 270000000;
////    float a_float = 2.0f;
////    double a_double = 2.0;
////    float* x_float, * y_float;
////    double* x_double, * y_double;
////    const int block_size = 104;
////    const int num_block = 96;
////
////    cudaMalloc(&x_float, n * sizeof(float));
////    cudaMalloc(&y_float, n * sizeof(float));
////    cudaMalloc(&x_double, n * sizeof(double));
////    cudaMalloc(&y_double, n * sizeof(double));
////
////    float* h_x_float = new float[n];
////    float* h_y_float = new float[n];
////    double* h_x_double = new double[n];
////    double* h_y_double = new double[n];
////
////    fill(n, h_x_float, h_y_float, h_x_double, h_y_double);
////    double start = omp_get_wtime();
////    saxpy(n, a_float, h_x_float, 1, h_y_float, 1);
////    double finish = omp_get_wtime();
////    std::cout << finish - start << std::endl;
////
////    fill(n, h_x_float, h_y_float,h_x_double, h_y_double);
////    cudaMemcpy(x_float, h_x_float, n * sizeof(float), cudaMemcpyHostToDevice);
////    cudaMemcpy(y_float, h_y_float, n * sizeof(float), cudaMemcpyHostToDevice);
////    cudaMemcpy(x_double, h_x_double, n * sizeof(double), cudaMemcpyHostToDevice);
////    cudaMemcpy(y_double, h_y_double, n * sizeof(double), cudaMemcpyHostToDevice);
////    start = omp_get_wtime();
////    saxpyGPU<< <num_block, block_size>> >(n, a_float, x_float, 1, y_float, 1);
////    cudaDeviceSynchronize();
////    finish = omp_get_wtime();
////    std::cout << finish - start << std::endl;
////
////    return 1;
////}