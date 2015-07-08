#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int eigs_sym_F77(MatrixXd &M, VectorXd &init_resid, int k, int m);
int eigs_gen_F77(MatrixXd &M, VectorXd &init_resid, int k, int m);
int eigs_sym_Cpp(MatrixXd &M, VectorXd &init_resid, int k, int m);
int eigs_gen_Cpp(MatrixXd &M, VectorXd &init_resid, int k, int m);

int main()
{
    srand(123);
    MatrixXd A = MatrixXd::Random(1000, 1000);
    A.array() -= 0.5;
    MatrixXd M = A.transpose() * A;

    VectorXd init_resid = VectorXd::Random(M.cols());
    init_resid.array() -= 0.5;

    int k = 10;
    int m = 30;

    clock_t t1, t2;

    t1 = clock();
    eigs_sym_F77(M, init_resid, k, m);
    t2 = clock();
    std::cout << "elapsed time for eigs_sym_F77: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";



    t1 = clock();
    eigs_sym_Cpp(M, init_resid, k, m);
    t2 = clock();
    std::cout << "elapsed time for eigs_sym_Cpp: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";



    t1 = clock();
    eigs_gen_F77(A, init_resid, k, m);
    t2 = clock();
    std::cout << "elapsed time for eigs_gen_F77: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";



    t1 = clock();
    eigs_gen_Cpp(A, init_resid, k, m);
    t2 = clock();
    std::cout << "elapsed time for eigs_gen_Cpp: "
              << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";

    return 0;
}
