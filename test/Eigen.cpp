// Test ../include/UpperHessenbergEigen.h
#include <UpperHessenbergEigen.h>
#include <Eigen/Eigenvalues>
#include <ctime>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;

TEST_CASE("Eigen decomposition of upper Hessenberg matrix", "[Eigen]")
{
    srand(123);
    int n = 100;
    MatrixXd m = MatrixXd::Random(n, n);
    m.array() -= 0.5;
    MatrixXd H = m.triangularView<Eigen::Upper>();
    H.diagonal(-1) = m.diagonal(-1);

    UpperHessenbergEigen<double> decomp(H);
    VectorXcd evals = decomp.eigenvalues();
    MatrixXcd evecs = decomp.eigenvectors();

    MatrixXcd err = H * evecs - evecs * evals.asDiagonal();

    INFO( "||HU - UD||_inf = " << err.cwiseAbs().maxCoeff() );
    REQUIRE( err.cwiseAbs().maxCoeff() == Approx(0.0) );

    clock_t t1, t2;
    t1 = clock();
    for(int i = 0; i < 100; i++)
    {
        UpperHessenbergEigen<double> decomp(H);
        VectorXcd evals = decomp.eigenvalues();
        MatrixXcd evecs = decomp.eigenvectors();
    }
    t2 = clock();
    std::cout << "elapsed time for UpperHessenbergEigen: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";

    t1 = clock();
    for(int i = 0; i < 100; i++)
    {
        Eigen::EigenSolver<MatrixXd> decomp(H);
        VectorXcd evals = decomp.eigenvalues();
        MatrixXcd evecs = decomp.eigenvectors();
    }
    t2 = clock();
    std::cout << "elapsed time for Eigen::EigenSolver: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";
}
