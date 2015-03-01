#ifndef MATOPDENSE_H
#define MATOPDENSE_H

#include <Eigen/Dense>

template <typename Scalar>
class MatOpDense:
    public MatOpWithTransProd<Scalar>,
    public MatOpWithComplexShiftSolve<Scalar>
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;

    MapMat mat;
    MapVec vec_x;
    MapVec vec_y;

    // shift solve for complex sigma
    virtual void complex_shift_solve(Scalar *x_in, Scalar *y_out) {}
public:
    MatOpDense(const Matrix &mat_) :
        MatOp<Scalar>(mat_.rows(), mat_.cols()),
        MatOpWithTransProd<Scalar>(mat_.rows(), mat_.cols()),
        MatOpWithComplexShiftSolve<Scalar>(mat_.rows(), mat_.cols()),
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        vec_x(NULL, 1),
        vec_y(NULL, 1)
    {}

    virtual ~MatOpDense() {}

    // y_out = A * x_in
    virtual void prod(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, MatOp<Scalar>::cols());
        new (&vec_y) MapVec(y_out, MatOp<Scalar>::rows());

        vec_y.noalias() = mat * vec_x;
    }

    // y_out = A' * x_in
    virtual void trans_prod(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, MatOp<Scalar>::rows());
        new (&vec_y) MapVec(y_out, MatOp<Scalar>::cols());

        vec_y.noalias() = mat.transpose() * vec_x;
    }

    // y_out = inv(A - sigma * I) * x_in
    virtual void shift_solve(Scalar *x_in, Scalar *y_out)
    {

    }
};


#endif // MATOPDENSE_H
