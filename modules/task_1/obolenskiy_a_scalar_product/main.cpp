// Copyright 2019 Obolenskiy Arseniy
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./scalar_product.h"

TEST(Scalar_Product_MPI, Test_On_Size_2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> v, u;
    const size_t vector_size = 2;
    if (rank == 0) {
        v = getRandomVector(vector_size);
        u = getRandomVector(vector_size);
    }

    int answer = getScalarProduct(v, u, vector_size);

    if (rank == 0) {
        int seqAnswer = 0;
        for (size_t i = 0; i < vector_size; ++i) {
            seqAnswer += v[i] * u[i];
        }
        ASSERT_EQ(seqAnswer, answer);
    }
}

TEST(Scalar_Product_MPI, Test_On_Size_20) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> v, u;
    const size_t vector_size = 20;

    if (rank == 0) {
        v = getRandomVector(vector_size);
        u = getRandomVector(vector_size);
    }

    int answer = getScalarProduct(v, u, vector_size);

    if (rank == 0) {
        int seqAnswer = 0;
        for (size_t i = 0; i < vector_size; ++i) {
            seqAnswer += v[i] * u[i];
        }
        ASSERT_EQ(seqAnswer, answer);
    }
}

TEST(Scalar_Product_MPI, Check_Whether_Vectors_Have_the_Same_Length) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> v, u;

    if (rank == 0) {
        v = getRandomVector(10);
        u = getRandomVector(20);
    }

    ASSERT_ANY_THROW(getScalarProduct(v, u, v.size()));
}

TEST(Scalar_Product_MPI, Check_Specified_Vector_Length) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> v, u;
    const size_t vector_size = 20;

    if (rank == 0) {
        v = getRandomVector(vector_size);
        u = getRandomVector(vector_size);
    }

    ASSERT_ANY_THROW(getScalarProduct(v, u, vector_size - 1));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
