#include <iostream>
#include <omp.h>

int min(int a, int b) {
    return (a < b) ? a : b;
}

template<typename Type>
void vecAdd(Type* arr, int num, Type* sum, Type init) {
    int rank = omp_get_thread_num();
    int count = omp_get_num_threads();
    int localNum = num / count + 1;
    int localStart = min(rank * localNum, num);
    int localEnd = min(localStart + localNum, num);
    Type localSum(init);

    for (int i = localStart; i < localEnd; i++) {
        localSum += arr[i];
    }

    #pragma omp critical
    *sum += localSum;
}

int main(int argc, char* argv[]) {
    int* arr = new int[5]{1, 2, 3, 4, 5};
    int sum = 0;
    int thread_count = strtol(argv[1], NULL, 10);

    #pragma omp parallel num_threads(thread_count)
    vecAdd<int>(arr, 5, &sum, 0);

    std::cout << sum << std::endl;
    if (sum == 15) {
        std::cout << "Right answer" << std::endl;
    } else {
        std::cout << "Wrong answer" << std::endl;
    }

    return 0;
}

/*
`using parallel for`

#pragma omp parallel for num_threads(thread_count) \
reduction(+: sum)
for (int i = 0; i < 5; i++) {
    sum += arr[i];
}
*/
