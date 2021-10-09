#include <iostream>
#include <omp.h>

int min(int a, int b) {
    return (a < b) ? a : b;
}

// matA: [M x K], matB: [K x N], matC: [M x N]
template<typename Type>
void matMul(Type** matA, Type** matB, Type** matC, int M, int N, int K, Type init) {
    int rank = omp_get_thread_num();
    int count = omp_get_num_threads();
    int localRow = M / count + 1;
    int localStart = min(rank * localRow, M);
    int localEnd = min(localStart + localRow, M);

    for (int i = localStart; i < localEnd; i++) {
        for (int j = 0; j < N; j++) {
            Type sum(init);
            for (int k = 0; k < K; k++) {
                sum += matA[i][k] * matB[k][j];
            }
            #pragma omp critical
            matC[i][j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    int** matA = new int*[2];
    matA[0] = new int[4]{1, 3, 2, 5};
    matA[1] = new int[4]{2, 4, 1, 3};

    int** matB = new int*[4];
    matB[0] = new int[3]{3, 1, 2};
    matB[1] = new int[3]{1, 2, 4};
    matB[2] = new int[3]{5, 3, 1};
    matB[3] = new int[3]{2, 6, 0};

    int** matC = new int*[2];
    matC[0] = new int[3]{0, 0, 0};
    matC[1] = new int[3]{0, 0, 0};

    int thread_count = strtol(argv[1], NULL, 10);

    #pragma omp parallel num_threads(thread_count)
    matMul<int>(matA, matB, matC, 2, 3, 4, 0);

    int** matCans = new int*[2];
    matCans[0] = new int[3]{26, 43, 16};
    matCans[1] = new int[3]{21, 31, 21};
    bool flag = true;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << matC[i][j] << ' ';
            if (matC[i][j] != matCans[i][j]) flag = false;
        }
        std::cout << std::endl;
    }

    if (flag) {
        std::cout << "Right answer" << std::endl;
    } else {
        std::cout << "Wrong answer" << std::endl;
    }

    return 0;
}


/*
`using parallel for`

#pragma omp parallel for num_threads(thread_count)
for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
        int sum = 0;
        for (int k = 0; k < 4; k++) {
            sum += matA[i][k] * matB[k][j];
        }
        matC[i][j] = sum;
    }
}
*/
