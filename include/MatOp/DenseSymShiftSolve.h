#ifndef DENSE_SYM_SHIFT_SOLVE_H
#define DENSE_SYM_SHIFT_SOLVE_H

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <stdexcept>

template <typename Scalar>
class DenseSymShiftSolve
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;

    const MapMat mat;
    const int dim_n;
    Eigen::LDLT<Matrix> solver;

public:
    DenseSymShiftSolve(Matrix &mat_) :
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        dim_n(mat_.rows())
    {
        if(mat_.rows() != mat_.cols())
            throw std::invalid_argument("DenseSymShiftSolve: matrix must be square");
    }

    int rows() { return dim_n; }
    int cols() { return dim_n; }

    // setting real sigma
    void set_shift(Scalar sigma)
    {
        solver.compute(mat - sigma * Matrix::Identity(dim_n, dim_n));
    }

    // y_out = inv(A - sigma * I) * x_in
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        MapVec x(x_in,  dim_n);
        MapVec y(y_out, dim_n);
        y.noalias() = solver.solve(x);
    }
};


#endif // DENSESYMSHIFTSOLVE_H
