// Copyright 2019 Obolenskiy Arseniy
#include <mpi.h>
#include <random>
#include <ctime>
#include <vector>
#include "../../../modules/task_1/obolenskiy_a_scalar_product/scalar_product.h"

static int offset = 0;

std::vector<int> getRandomVector(int sz) {
    std::mt19937 gen;
    gen.seed(time(0) + ++offset);
    std::vector<int> vec(sz);
    for (int i = 0; i < sz; ++i)
        vec[i] = gen() % 100;
    return vec;
}

int getScalarProduct(std::vector <int> a, std::vector <int> b, int vector_size) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int delta = vector_size / size;
    const int rem = vector_size % size;

    if (rank == 0) {
        for (int proc = 1; proc < size; ++proc) {
            MPI_Send(&a[rem] + proc * delta, delta, MPI_INT, proc, 1, MPI_COMM_WORLD);
            MPI_Send(&b[rem] + proc * delta, delta, MPI_INT, proc, 2, MPI_COMM_WORLD);
        }
    }

    std::vector <int> c(delta), d(delta);

    if (rank == 0) {
        c.resize(rem + delta);
        d.resize(rem + delta);
        c = std::vector<int>(a.begin(), a.begin() + rem + delta);
        d = std::vector<int>(b.begin(), b.begin() + rem + delta);
    } else {
        MPI_Status status;
        MPI_Recv(&c[0], delta, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&d[0], delta, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
    }

    int ans = 0;
    for (unsigned i = 0; i < c.size(); ++i) {
        ans += c[i] * d[i];
    }

    if (rank == 0) {
        for (int proc = 1; proc < size; ++proc) {
            int temp;
            MPI_Status status;
            MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
            ans += temp;
        }
    } else {
        MPI_Send(&ans, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }

    return ans;
}
