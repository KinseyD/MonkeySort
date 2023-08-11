#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

int array[] = { 22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 };
const int arraySize = sizeof(array) / sizeof(int);

// 排序
__global__ void sortWithCuda(curandState* states, int** darray, bool* flag)
{
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned x = curand(&states[idx]) % arraySize;
	unsigned y = curand(&states[idx]) % arraySize;
	if (x == y)
		return;
	darray[threadIdx.x][x] ^= darray[threadIdx.x][y];
	darray[threadIdx.x][y] ^= darray[threadIdx.x][x];
	darray[threadIdx.x][x] ^= darray[threadIdx.x][y];
	for (int i = 1; i < arraySize; i++)
	{
		if (darray[threadIdx.x][i] < darray[threadIdx.x][i - 1])
			return;
	}
	printf("线程(%u,%u)已完成\n", blockIdx.x, threadIdx.x);
	*flag = true;
}

__global__ void randInit(curandState* states)
{
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(blockIdx.x, threadIdx.x, 0, &states[idx]);
}

int main()
{
	size_t blk = 1e4;
	size_t trd = 32;

	// 在显存中开辟二维数组
	int** deviceArray;
	int** midArray;
	cudaMalloc((void***)&deviceArray, trd * sizeof(int*));
	midArray = (int**)malloc(trd * sizeof(int*));
	for (int i = 0; i < trd; i++)
	{
		cudaMalloc((void**)&midArray[i], arraySize * sizeof(int));
		cudaMemcpy(midArray[i], array, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	}
	cudaMemcpy(deviceArray, midArray, trd * sizeof(int*), cudaMemcpyHostToDevice);
	free(midArray);

	// 初始化随机函数
	curandState* states;
	cudaMalloc(&states, blk * trd * sizeof(curandState));
	randInit << <blk, trd >> > (states);
	cudaDeviceSynchronize();

	// 记录是否完成
	bool* dFlag = nullptr;
	cudaMalloc(&dFlag, sizeof(bool));
	bool hFlag = false;
	cudaMemcpy(dFlag, &hFlag, sizeof(bool), cudaMemcpyHostToDevice);

	printf("正在为%d个元素进行排序...\n\n", arraySize);

	// 准备计时
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	size_t rounds = 0;
	cudaEventRecord(start);
	while (!hFlag)
	{
		sortWithCuda << <blk, trd >> > (states, deviceArray, dFlag);
		cudaMemcpy(&hFlag, dFlag, sizeof(bool), cudaMemcpyDeviceToHost);
		++rounds;
	}
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float cost;
	cudaEventElapsedTime(&cost, start, end);
	if (cost > 1e4)
		printf("\n共有%llu个线程参与，耗时%.3fs\n", blk * trd * rounds, cost / 1000);
	else
		printf("\n共有%llu个线程参与，耗时%.2fms\n", blk * trd * rounds, cost);

	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cudaFree(deviceArray);
	cudaFree(states);

	return 0;
}