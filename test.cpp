#include <SymEigsSolver.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
    MatrixXd A = MatrixXd::Random(10, 10);
    A = A.adjoint() * A;

    int k = 2;
    int m = 6;

    Eigen::SelfAdjointEigenSolver<MatrixXd> eig(A);
    std::cout << "true eigenvalues = \n" << eig.eigenvalues() << "\n\n";

    DenseMatOp<double> op(A);
    SymEigsSolver<double> eigs(&op, k, m);
    int niter = eigs.compute();

    VectorXd evals = eigs.eigenvalues();
    MatrixXd evecs = eigs.eigenvectors();

    std::cout << "computed eigenvalues D = \n" << evals << "\n\n";
    std::cout << "computed eigenvectors U = \n" << evecs << "\n\n";
    std::cout << "AU - UD = \n" << A * evecs - evecs * evals.asDiagonal() << "\n\n";
    std::cout << "niter = " << niter << std::endl;

    return 0;
}
