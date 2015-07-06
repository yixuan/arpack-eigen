#include <Eigen/Dense>
#include <iostream>

#include <SymEigsSolver.h>
#include <MatOp/DenseSymShiftSolve.h>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

template <int SelectionRule>
void run_test(const Matrix &A, int k, int m, double sigma)
{
    Matrix mat;
    if(SelectionRule == BOTH_ENDS)
    {
        mat = A.adjoint() + A;
    } else {
        mat = A.adjoint() * A;
    }

    // Eigen::SelfAdjointEigenSolver<MatrixXd> eig(mat);
    // std::cout << "all eigenvalues = \n" << eig.eigenvalues().transpose() << "\n";

    DenseSymShiftSolve<double> op(mat);
    SymEigsShiftSolver<double, SelectionRule, DenseSymShiftSolve<double>> eigs(&op, k, m, sigma);
    eigs.init();
    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    REQUIRE( nconv > 0 );

    Vector evals = eigs.eigenvalues();
    Matrix evecs = eigs.eigenvectors();

    // evals.print("computed eigenvalues D =");
    // evecs.print("computed eigenvectors U =");
    Matrix err = mat * evecs - evecs * evals.asDiagonal();

    INFO( "nconv = " << nconv );
    INFO( "niter = " << niter );
    INFO( "nops = " << nops );
    INFO( "||AU - UD||_inf = " << err.array().abs().maxCoeff() );
    REQUIRE( err.array().abs().maxCoeff() == Approx(0.0) );
}

TEST_CASE("Eigensolver of symmetric real matrix in shift-and-invert mode", "[eigs_sym_shift]")
{
    srand(123);
    Matrix A = Eigen::MatrixXd::Random(10, 10);

    int k = 3;
    int m = 6;
    double sigma = 1.0;

    SECTION( "Largest Magnitude" )
    {
        run_test<LARGEST_MAGN>(A, k, m, sigma);
    }
    SECTION( "Largest Value" )
    {
        run_test<LARGEST_ALGE>(A, k, m, sigma);
    }
    SECTION( "Smallest Magnitude" )
    {
        run_test<SMALLEST_MAGN>(A, k, m, sigma);
    }
    SECTION( "Smallest Value" )
    {
        run_test<SMALLEST_ALGE>(A, k, m, sigma);
    }
    SECTION( "Both Ends" )
    {
        run_test<BOTH_ENDS>(A, k, m, sigma);
    }
}
