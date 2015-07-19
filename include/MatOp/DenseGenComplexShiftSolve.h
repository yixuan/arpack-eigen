#ifndef DENSE_GEN_COMPLEX_SHIFT_SOLVE_H
#define DENSE_GEN_COMPLEX_SHIFT_SOLVE_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <stdexcept>

template <typename Scalar>
class DenseGenComplexShiftSolve
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;

    typedef std::complex<Scalar> Complex;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMatrix;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVector;
    typedef Eigen::PartialPivLU<ComplexMatrix> ComplexSolver;

    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    const MapMat mat;
    const int dim_n;
    ComplexSolver solver;
    ComplexVector x_cache;

public:
    DenseGenComplexShiftSolve(ConstGenericMatrix &mat_) :
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        dim_n(mat_.rows())
    {
        if(mat_.rows() != mat_.cols())
            throw std::invalid_argument("DenseGenComplexShiftSolve: matrix must be square");
    }

    int rows() { return dim_n; }
    int cols() { return dim_n; }

    // setting complex sigma
    void set_shift(Scalar sigmar, Scalar sigmai)
    {
        ComplexMatrix cmat = mat.template cast<Complex>();
        cmat.diagonal().array() -= Complex(sigmar, sigmai);
        solver.compute(cmat);
        x_cache.resize(dim_n);
        x_cache.setZero();
    }

    // y_out = Re( inv(A - sigma * I) * x_in )
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        x_cache.real() = MapVec(x_in, dim_n);
        MapVec y(y_out, dim_n);
        y.noalias() = solver.solve(x_cache).real();
    }
};


#endif // DENSE_GEN_COMPLEX_SHIFT_SOLVE_H
