#include <Eigen/Dense>
#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;

int eigs_sym_Cpp(MatrixXd &M, VectorXd &init_resid, int k, int m)
{
    DenseGenMatProd<double> op(M);
    SymEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.data());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    VectorXd evals = eigs.eigenvalues();
    MatrixXd evecs = eigs.eigenvectors();

    std::cout << "computed eigenvalues D = \n" << evals.transpose() << std::endl;
    std::cout << "first 5 rows of computed eigenvectors U = \n" << evecs.topRows<5>() << std::endl;
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    return 0;
}



int eigs_gen_Cpp(MatrixXd &M, VectorXd &init_resid, int k, int m)
{
    DenseGenMatProd<double> op(M);
    GenEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.data());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    VectorXcd evals = eigs.eigenvalues();
    MatrixXcd evecs = eigs.eigenvectors();

    std::cout << "computed eigenvalues D = \n" << evals.transpose() << std::endl;
    std::cout << "first 5 rows of computed eigenvectors U = \n" << evecs.topRows<5>() << std::endl;
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    return 0;
}
