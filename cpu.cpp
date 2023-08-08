#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>

bool isSorted(int arr[], int size) {
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

void monkeySort(int arr[], int size, long long int& iterations) {
    std::srand(std::time(nullptr));  // 使用当前时间作为随机数生成器的种子

    while (!isSorted(arr, size)) {
        // 生成两个随机索引
        int index1 = std::rand() % size;
        int index2 = std::rand() % size;

        // 交换两个索引处的元素
        std::swap(arr[index1], arr[index2]);

        iterations++;
    }
}

int main() {
    int arr[] = { 14,13,12,11,10,9,8,7,6,5,4,3,2,1 };  // 待排序的数组
    int size = sizeof(arr) / sizeof(arr[0]);

    long long int iterations = 0;  // 记录执行次数

    std::clock_t start = std::clock();  // 记录开始时间

    monkeySort(arr, size, iterations);

    std::clock_t end = std::clock();  // 记录结束时间
    double duration = static_cast<double>(end - start) / CLOCKS_PER_SEC;  // 计算耗时

    std::cout << "排序后的数组：";
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "执行次数：" << iterations << std::endl;
    std::cout << "花费时间：" << duration << " 秒" << std::endl;

    return 0;
}