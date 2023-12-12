#pragma once
#include <cuda_runtime.h>
__global__ void saxpyGPU(int n, float a, float* x, int incx, float* y, int incy);
__global__ void daxpyGPU(int n, double a, double* x, int incx, double* y, int incy);