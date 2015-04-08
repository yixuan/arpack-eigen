#include <iostream>
#include <SymEigsSolver.h>
#include <MatOpDense.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

template <int SelectionRule>
void run_test(const MatrixXd &A, int k, int m, double sigma)
{
    MatrixXd mat;
    if(SelectionRule == BOTH_ENDS)
    {
        mat = A.adjoint() + A;
    } else {
        mat = A.adjoint() * A;
    }

    Eigen::SelfAdjointEigenSolver<MatrixXd> eig(mat);
    std::cout << "all eigenvalues = \n" << eig.eigenvalues().transpose() << "\n";

    MatOpDense<double> op(mat);
    SymEigsShiftSolver<double, SelectionRule> eigs(&op, k, m, sigma);
    eigs.init();
    int nconv = eigs.compute();
    int niter, nops;
    eigs.info(niter, nops);

    if(nconv > 0)
    {
        VectorXd evals = eigs.eigenvalues();
        MatrixXd evecs = eigs.eigenvectors();

        std::cout << "computed eigenvalues D = \n" << evals.transpose() << "\n";
        //std::cout << "computed eigenvectors U = \n" << evecs << "\n\n";
        std::cout << "||AU - UD||_inf = " << (mat * evecs - evecs * evals.asDiagonal()).array().abs().maxCoeff() << "\n";
    }
    std::cout << "nconv = " << nconv << "\n";
    std::cout << "niter = " << niter << "\n";
    std::cout << "nops = " << nops << "\n";
}

int main()
{
    srand(123);
    MatrixXd A = MatrixXd::Random(10, 10);

    int k = 3;
    int m = 6;
    double sigma = 1.0;

    std::cout << "===== Largest Magnitude =====\n";
    run_test<LARGEST_MAGN>(A, k, m, sigma);

    std::cout << "\n===== Largest Value =====\n";
    run_test<LARGEST_ALGE>(A, k, m, sigma);

    std::cout << "\n===== Smallest Magnitude =====\n";
    run_test<SMALLEST_MAGN>(A, k, m, sigma);

    std::cout << "\n===== Smallest Value =====\n";
    run_test<SMALLEST_ALGE>(A, k, m, sigma);

    std::cout << "\n===== Both Ends =====\n";
    run_test<BOTH_ENDS>(A, k, m, sigma);

    return 0;
}
