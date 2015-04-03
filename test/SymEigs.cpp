#include <iostream>
#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <MatOpDense.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

template <int SelectionRule>
void test(const MatrixXd &A, int k, int m)
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
    SymEigsSolver<double, SelectionRule> eigs(&op, k, m);
    eigs.init();
    int niter = eigs.compute();
    int nops;
    eigs.info(nops);

    VectorXd evals = eigs.eigenvalues();
    MatrixXd evecs = eigs.eigenvectors();

    std::cout << "computed eigenvalues D = \n" << evals.transpose() << "\n";
    //std::cout << "computed eigenvectors U = \n" << evecs << "\n\n";
    std::cout << "||AU - UD||_inf = " << (mat * evecs - evecs * evals.asDiagonal()).array().abs().maxCoeff() << "\n";
    std::cout << "niter = " << niter << "\n";
    std::cout << "nops = " << nops << "\n";
}

int main()
{
    srand(123);
    MatrixXd A = MatrixXd::Random(10, 10);

    int k = 3;
    int m = 6;

    std::cout << "===== Largest Magnitude =====\n";
    test<LARGEST_MAGN>(A, k, m);

    std::cout << "\n===== Largest Value =====\n";
    test<LARGEST_ALGE>(A, k, m);

    std::cout << "\n===== Smallest Value =====\n";
    test<SMALLEST_MAGN>(A, k, m);

    std::cout << "\n===== Smallest Value =====\n";
    test<SMALLEST_ALGE>(A, k, m);

    std::cout << "\n===== Both Ends =====\n";
    test<BOTH_ENDS>(A, k, m);

    return 0;
}
