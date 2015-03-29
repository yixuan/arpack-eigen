#include <iostream>
#include <SymEigsSolver.h>
#include <MatOpDense.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int run_Cpp(MatrixXd &M, int k, int m)
{
    MatOpDense<double> op(M);
    SymEigsSolver<double, LARGEST_MAGN> eigs(&op, k, m);
    eigs.init();
    int niter = eigs.compute();

    VectorXd evals = eigs.eigenvalues();
    MatrixXd evecs = eigs.eigenvectors();

    std::cout << "computed eigenvalues D = \n" << evals.transpose() << std::endl;
    std::cout << "first 5 rows of computed eigenvectors U = \n" << evecs.topLeftCorner(5, k) << std::endl;
    std::cout << "niter = " << niter << std::endl;

    return 0;
}
