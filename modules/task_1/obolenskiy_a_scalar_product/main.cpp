// Copyright 2019 Obolenskiy Arseniy
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <algorithm>
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

    int64_t answer = getScalarProduct(v, u, vector_size);

    if (rank == 0) {
        int64_t seqAnswer = 0;
        for (size_t i = 0; i < vector_size; ++i) {
            seqAnswer += (int64_t)v[i] * u[i];
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

    int64_t answer = getScalarProduct(v, u, vector_size);

    if (rank == 0) {
        int64_t seqAnswer = 0;
        for (size_t i = 0; i < vector_size; ++i) {
            seqAnswer += (int64_t)v[i] * u[i];
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

TEST(Scalar_Product_MPI, Check_Empty_Vector_Handling) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> v, u;
    v = getRandomVector(0);
    u = getRandomVector(0);

    int64_t answer = getScalarProduct(v, u, 0);

    if (rank == 0) {
        ASSERT_EQ(0, answer);
    }
}

TEST(Scalar_Product_MPI, Reversed_Vectors_Give_The_Same_Answer) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> v, u;
    const size_t vector_size = 2;
    if (rank == 0) {
        v = getRandomVector(vector_size);
        u = getRandomVector(vector_size);
    }

    int64_t answer = getScalarProduct(v, u, vector_size);

    if (rank == 0) {
        std::reverse(v.begin(), v.end());
        std::reverse(u.begin(), u.end());
    }

    int64_t answerReversed = getScalarProduct(v, u, vector_size);

    if (rank == 0) {
        ASSERT_EQ(answer, answerReversed);
    }
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
