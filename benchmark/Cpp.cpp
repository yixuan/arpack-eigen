#include <Eigen/Core>
#include <iostream>
#include <ctime>

#include <SymEigsSolver.h>
#include <GenEigsSolver.h>
#include <MatOp/DenseGenMatProd.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;

void eigs_sym_Cpp(MatrixXd &M, VectorXd &init_resid, int k, int m,
                  double &time_used, double &prec_err)
{
    clock_t start, end;
    start = clock();

    DenseGenMatProd<double> op(M);
    SymEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.data());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    VectorXd evals = eigs.eigenvalues();
    MatrixXd evecs = eigs.eigenvectors();

/*
    std::cout << "computed eigenvalues D = \n" << evals.transpose() << std::endl;
    std::cout << "first 5 rows of computed eigenvectors U = \n" << evecs.topRows<5>() << std::endl;
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;
*/

    end = clock();
    time_used = (end - start) / double(CLOCKS_PER_SEC) * 1000;

    MatrixXd err = M * evecs - evecs * evals.asDiagonal();
    prec_err = err.cwiseAbs().maxCoeff();
}



void eigs_gen_Cpp(MatrixXd &M, VectorXd &init_resid, int k, int m,
                  double &time_used, double &prec_err)
{
    clock_t start, end;
    start = clock();

    DenseGenMatProd<double> op(M);
    GenEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init(init_resid.data());

    int nconv = eigs.compute();
    int niter = eigs.num_iterations();
    int nops = eigs.num_operations();

    VectorXcd evals = eigs.eigenvalues();
    MatrixXcd evecs = eigs.eigenvectors();

/*
    std::cout << "computed eigenvalues D = \n" << evals.transpose() << std::endl;
    std::cout << "first 5 rows of computed eigenvectors U = \n" << evecs.topRows<5>() << std::endl;
    std::cout << "nconv = " << nconv << std::endl;
    std::cout << "niter = " << niter << std::endl;
    std::cout << "nops = " << nops << std::endl;

    MatrixXcd err = M * evecs - evecs * evals.asDiagonal();
    std::cout << "||AU - UD||_inf = " << err.array().abs().maxCoeff() << std::endl;
*/

    end = clock();
    time_used = (end - start) / double(CLOCKS_PER_SEC) * 1000;

    MatrixXcd err = M * evecs - evecs * evals.asDiagonal();
    prec_err = err.cwiseAbs().maxCoeff();
}
