#ifndef DENSE_GEN_MAT_PROD_H
#define DENSE_GEN_MAT_PROD_H

#include <Eigen/Core>

template <typename Scalar>
class DenseGenMatProd
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;

    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    const MapMat mat;

public:
    DenseGenMatProd(ConstGenericMatrix &mat_) :
        mat(mat_.data(), mat_.rows(), mat_.cols())
    {}

    int rows() { return mat.rows(); }
    int cols() { return mat.cols(); }

    // y_out = A * x_in
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        MapVec x(x_in, mat.cols());
        MapVec y(y_out, mat.rows());
        y = mat * x;
    }
};


#endif // DENSE_GEN_MAT_PROD_H
