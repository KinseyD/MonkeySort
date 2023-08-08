#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

int array[] = { 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 };
const int arraySize = sizeof(array) / sizeof(int);

__device__ clock_t cost;

// 排序
__global__ void sortWithCuda(curandState* states, int** darray, size_t size)
{
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned x = curand(&states[idx]) % arraySize;
	unsigned y = curand(&states[idx]) % arraySize;
	// printf("(%u %u), ", x, y);
	if (x == y)
		return;
	darray[threadIdx.x][x] ^= darray[threadIdx.x][y];
	darray[threadIdx.x][y] ^= darray[threadIdx.x][x];
	darray[threadIdx.x][x] ^= darray[threadIdx.x][y];
	bool flag = true;
	for (int i = 1; i < arraySize; i++)
	{
		// printf("%d,", darray[threadIdx.x][i]);
		if (darray[threadIdx.x][i] < darray[threadIdx.x][i - 1])
			flag = false;
	}
	if (flag)
	{
		printf("线程(%u,%u)已完成，耗时%d\n", blockIdx.x, threadIdx.x, (clock() - cost) / CLK_TCK);
	}
}

// 设置开始时间
__global__ void setClock()
{
	cost = clock();
}

__global__ void randInit(curandState* states, size_t size)
{
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(blockIdx.x, threadIdx.x, 0, &states[idx]);
}

__global__ void debug(int** array)
{
	for (size_t i = 0; i < arraySize; i++)
	{
		printf("%d ", array[32][i]);
	}
}

int main()
{
	size_t blk = 4e6;
	size_t trd = 32;

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

	curandState* states;
	cudaMalloc(&states, blk * trd * sizeof(curandState));

	randInit << <blk, trd >> > (states, blk * trd);
	cudaDeviceSynchronize();

	printf("猴子开始工作！\n");
	setClock << <1, 1 >> > ();
	cudaDeviceSynchronize();

	sortWithCuda << <blk, trd >> > (states, deviceArray, blk * trd);
	// cudaDeviceSynchronize();

	return 0;
}