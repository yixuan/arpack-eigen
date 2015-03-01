#include <SymEigsSolver.h>
#include <iostream>

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
    std::cout << "true eigenvalues = \n" << eig.eigenvalues().transpose() << "\n\n";

    DenseMatOp<double> op(mat);
    SymEigsSolver<double, SelectionRule> eigs(&op, k, m);
    eigs.init();
    int niter = eigs.compute();

    VectorXd evals = eigs.eigenvalues();
    MatrixXd evecs = eigs.eigenvectors();

    std::cout << "computed eigenvalues D = \n" << evals.transpose() << "\n\n";
    std::cout << "computed eigenvectors U = \n" << evecs << "\n\n";
    std::cout << "AU - UD = \n" << mat * evecs - evecs * evals.asDiagonal() << "\n\n";
    std::cout << "niter = " << niter << std::endl;
}

int main()
{
    MatrixXd A = MatrixXd::Random(10, 10);

    int k = 3;
    int m = 6;

    test<LARGEST_MAGN>(A, k, m);
    test<LARGEST_ALGE>(A, k, m);
    test<SMALLEST_MAGN>(A, k, m);
    test<SMALLEST_ALGE>(A, k, m);
    test<BOTH_ENDS>(A, k, m);

    return 0;
}
