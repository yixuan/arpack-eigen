#include <SymEigsSolver.h>
#include <iostream>

using Eigen::MatrixXd;

int main()
{
    MatrixXd A = MatrixXd::Random(10, 10);
    A = A.adjoint() * A;

    int k = 2;
    int m = 6;

    Eigen::SelfAdjointEigenSolver<MatrixXd> eig(A);
    std::cout << "true eigenvalues = \n" << eig.eigenvalues() << std::endl;

    SymEigsSolver eigs(A, k, m);
    int niter = eigs.compute();

    std::cout << "niter = " << niter << std::endl;
    std::cout << "computed eigenvalues = \n" << eigs.eigenvalues() << std::endl;

    return 0;
}
