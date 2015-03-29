// Test ../include/UpperHessenbergQR.h
#include <iostream>
#include <UpperHessenbergQR.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

void QR_UpperHessenberg()
{
    srand(123);
    int n = 10;
    MatrixXd m = MatrixXd::Random(n, n);
    m.array() -= 0.5;
    MatrixXd H = m.triangularView<Eigen::Upper>();
    H.diagonal(-1) = m.diagonal(-1);

    UpperHessenbergQR<double> decomp(H);

    // Obtain Q matrix
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd Q = I;
    decomp.applyQY(Q);

    // Test orthogonality
    MatrixXd QtQ = Q.transpose() * Q;
    std::cout << "||Q'Q - I||_inf = " << (QtQ - I).cwiseAbs().maxCoeff() << std::endl;

    MatrixXd QQt = Q * Q.transpose();
    std::cout << "||QQ' - I||_inf = " << (QQt - I).cwiseAbs().maxCoeff() << std::endl;

    // Calculate R = Q'H
    MatrixXd R = H;
    decomp.applyQtY(R);
    MatrixXd Rlower = R.triangularView<Eigen::Lower>();
    Rlower.diagonal().setZero();
    std::cout << "whether R is upper triangular, error = "
              << Rlower.cwiseAbs().maxCoeff() << std::endl;

    // Compare H and QR
    std::cout << "||H - QR||_inf = " << (H - Q * R).cwiseAbs().maxCoeff() << std::endl;

    // Testing "apply" functions
    MatrixXd Y = MatrixXd::Random(n, n);

    MatrixXd QY = Y;
    decomp.applyQY(QY);
    std::cout << "max error of QY = " << (QY - Q * Y).cwiseAbs().maxCoeff() << std::endl;

    MatrixXd YQ = Y;
    decomp.applyYQ(YQ);
    std::cout << "max error of YQ = " << (YQ - Y * Q).cwiseAbs().maxCoeff() << std::endl;

    MatrixXd QtY = Y;
    decomp.applyQtY(QtY);
    std::cout << "max error of Q'Y = " << (QtY - Q.transpose() * Y).cwiseAbs().maxCoeff() << std::endl;

    MatrixXd YQt = Y;
    decomp.applyYQt(YQt);
    std::cout << "max error of YQ' = " << (YQt - Y * Q.transpose()).cwiseAbs().maxCoeff() << std::endl;
}

void QR_Tridiagonal()
{
    srand(123);
    int n = 100;
    MatrixXd m = MatrixXd::Random(n, n);
    m.array() -= 0.5;
    MatrixXd H = MatrixXd::Zero(n, n);
    H.diagonal() = m.diagonal();
    H.diagonal(-1) = m.diagonal(-1);
    H.diagonal(1) = m.diagonal(-1);

    TridiagQR<double> decomp(H);

    // Obtain Q matrix
    MatrixXd I = MatrixXd::Identity(n, n);
    MatrixXd Q = I;
    decomp.applyQY(Q);

    // Test orthogonality
    MatrixXd QtQ = Q.transpose() * Q;
    std::cout << "||Q'Q - I||_inf = " << (QtQ - I).cwiseAbs().maxCoeff() << std::endl;

    MatrixXd QQt = Q * Q.transpose();
    std::cout << "||QQ' - I||_inf = " << (QQt - I).cwiseAbs().maxCoeff() << std::endl;

    // Calculate R = Q'H
    MatrixXd R = H;
    decomp.applyQtY(R);
    MatrixXd Rlower = R.triangularView<Eigen::Lower>();
    Rlower.diagonal().setZero();
    std::cout << "whether R is upper triangular, error = "
              << Rlower.cwiseAbs().maxCoeff() << std::endl;

    // Compare H and QR
    std::cout << "||H - QR||_inf = " << (H - Q * R).cwiseAbs().maxCoeff() << std::endl;

    // Testing "apply" functions
    MatrixXd Y = MatrixXd::Random(n, n);

    MatrixXd QY = Y;
    decomp.applyQY(QY);
    std::cout << "max error of QY = " << (QY - Q * Y).cwiseAbs().maxCoeff() << std::endl;

    MatrixXd YQ = Y;
    decomp.applyYQ(YQ);
    std::cout << "max error of YQ = " << (YQ - Y * Q).cwiseAbs().maxCoeff() << std::endl;

    MatrixXd QtY = Y;
    decomp.applyQtY(QtY);
    std::cout << "max error of Q'Y = " << (QtY - Q.transpose() * Y).cwiseAbs().maxCoeff() << std::endl;

    MatrixXd YQt = Y;
    decomp.applyYQt(YQt);
    std::cout << "max error of YQ' = " << (YQt - Y * Q.transpose()).cwiseAbs().maxCoeff() << std::endl;
}

int main()
{
    std::cout << "========== Test of upper Hessenberg matrix ==========\n";
    QR_UpperHessenberg();

    std::cout << "\n========== Test of Tridiagonal matrix ==========\n";
    QR_Tridiagonal();

}
