#ifndef MATOPDENSE_H
#define MATOPDENSE_H

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <limits>

template <typename Scalar>
class MatOpDense:
    public MatOpWithTransProd<Scalar>,
    public MatOpWithComplexShiftSolve<Scalar>
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;
    typedef Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, Eigen::Dynamic> ComplexMatrix;
    typedef Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> ComplexVector;
    typedef Eigen::PartialPivLU<Matrix> RealSolver;
    typedef Eigen::PartialPivLU<Eigen::MatrixXcd> ComplexSolver;

    const MapMat mat;
    MapVec vec_x;
    MapVec vec_y;
    ComplexVector vec_cx;

    RealSolver rsolver;
    ComplexSolver csolver;

    bool sigma_is_real;

    // shift solve for real sigma
    virtual void real_shift_solve(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, mat.cols());
        new (&vec_y) MapVec(y_out, mat.rows());

        vec_y.noalias() = rsolver.solve(vec_x);
    }

    // shift solve for complex sigma
    virtual void complex_shift_solve(Scalar *x_in, Scalar *y_out)
    {
        vec_cx.real() = MapVec(x_in, mat.cols());
        new (&vec_y) MapVec(y_out, mat.rows());

        vec_y.noalias() = csolver.solve(vec_cx).real();
    }
public:
    MatOpDense(const Matrix &mat_) :
        MatOp<Scalar>(mat_.rows(), mat_.cols()),
        MatOpWithTransProd<Scalar>(mat_.rows(), mat_.cols()),
        MatOpWithComplexShiftSolve<Scalar>(mat_.rows(), mat_.cols()),
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        vec_x(NULL, 1),
        vec_y(NULL, 1),
        sigma_is_real(false)
    {}

    virtual ~MatOpDense() {}

    // y_out = A * x_in
    virtual void prod(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, mat.cols());
        new (&vec_y) MapVec(y_out, mat.rows());

        vec_y.noalias() = mat * vec_x;
    }

    // y_out = A' * x_in
    virtual void trans_prod(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, mat.rows());
        new (&vec_y) MapVec(y_out, mat.cols());

        vec_y.noalias() = mat.transpose() * vec_x;
    }

    // setting complex shift
    virtual void set_shift(Scalar sigmar, Scalar sigmai)
    {
        if(std::abs(sigmai) < std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {
            rsolver.compute(mat - sigmar * Matrix::Identity(mat.rows(), mat.cols()));
            sigma_is_real = true;
        } else {
            ComplexMatrix cmat = mat.template cast< std::complex<Scalar> >();
            cmat.diagonal().array() -= std::complex<Scalar>(sigmar, sigmai);
            csolver.compute(cmat);
            sigma_is_real = false;
            vec_cx.resize(mat.cols());
            vec_cx.setZero();
        }
    }

    // y_out = inv(A - sigma * I) * x_in
    virtual void shift_solve(Scalar *x_in, Scalar *y_out)
    {
        if(sigma_is_real)
            real_shift_solve(x_in, y_out);
        else
            complex_shift_solve(x_in, y_out);
    }
};


#endif // MATOPDENSE_H
