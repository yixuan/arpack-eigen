#include <iostream>
#include <SymEigsSolver.h>
#include <MatOpDense.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int run_Cpp(MatrixXd &M, VectorXd &init_resid, int k, int m)
{
    MatOpDense<double> op(M);
    SymEigsSolver<double, LARGEST_MAGN> eigs(&op, k, m);
    eigs.init(init_resid.data());
    int nconv = eigs.compute();
    int niter, nops;
    eigs.info(niter, nops);

    VectorXd evals = eigs.eigenvalues();
    MatrixXd evecs = eigs.eigenvectors();

    std::cout << "computed eigenvalues D = \n" << evals.transpose() << std::endl;
    std::cout << "first 5 rows of computed eigenvectors U = \n" << evecs.topLeftCorner(5, k) << std::endl;
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    return 0;
}
