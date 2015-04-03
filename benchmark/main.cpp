#include <iostream>
#include <SymEigsSolver.h>
#include <MatOpDense.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int run_F77(MatrixXd &M, VectorXd &init_resid, int k, int m);
int run_Cpp(MatrixXd &M, VectorXd &init_resid, int k, int m);

int main()
{
    srand(123);
    MatrixXd A = MatrixXd::Random(1000, 1000);
    A.array() -= 0.5;
    MatrixXd M = A.transpose() * A;

    VectorXd init_resid = VectorXd::Random(M.cols());
    init_resid.array() -= 0.5;

    int k = 10;
    int m = 20;

    clock_t t1, t2;
    t1 = clock();

    run_F77(M, init_resid, k, m);

    t2 = clock();
    std::cout << "elapsed time for F77 version: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";

    t1 = clock();

    run_Cpp(M, init_resid, k, m);

    t2 = clock();
    std::cout << "elapsed time for C++ version: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";

    return 0;
}
