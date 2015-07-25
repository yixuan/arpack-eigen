#ifndef DOUBLE_SHIFT_QR_H
#define DOUBLE_SHIFT_QR_H

#include <Eigen/Core>
#include <stdexcept>

template <typename Scalar = double>
class DoubleShiftQR
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Matrix3X;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
    typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
    typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

    typedef Eigen::Ref<Matrix> GenericMatrix;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    int n;
    Matrix mat_H;
    // Householder reflectors
    Matrix3X ref_u;

    bool computed;

    void compute_reflector(const Scalar &x1, const Scalar &x2, const Scalar &x3, int ind)
    {
        Scalar tmp = x2 * x2 + x3 * x3;
        // x1' = x1 - rho * ||x||
        // rho = -sign(x1)
        Scalar x1_new = x1 - ((x1 <= 0) - (x1 > 0)) * std::sqrt(x1 * x1 + tmp);
        Scalar x_norm = std::sqrt(x1_new * x1_new + tmp);
        ref_u(0, ind) = x1_new / x_norm;
        ref_u(1, ind) = x2 / x_norm;
        ref_u(2, ind) = x3 / x_norm;
    }

    void compute_reflector(const Scalar *x, int ind)
    {
        compute_reflector(x[0], x[1], x[2], ind);
    }

    // P = I - 2 * u * u' = P'
    void apply_PX(GenericMatrix X, int u_ind)
    {
        if(u_ind == n - 2)
        {
            Vector2 u;
            u[0] = ref_u(0, u_ind);
            u[1] = ref_u(1, u_ind);
            X -= (2 * u) * (u.transpose() * X);
        } else {
            Vector3 u = ref_u.col(u_ind);
            X -= (2 * u) * (u.transpose() * X);
        }
    }

    void apply_XP(GenericMatrix X, int u_ind)
    {
        if(u_ind == n - 2)
        {
            Vector2 u;
            u[0] = ref_u(0, u_ind);
            u[1] = ref_u(1, u_ind);
            X -= (X * (2 * u)) * u.transpose();
        } else {
            Vector3 u = ref_u.col(u_ind);
            X -= (X * (2 * u)) * u.transpose();
        }
    }
public:
    DoubleShiftQR() :
        n(0), computed(false)
    {}

    DoubleShiftQR(ConstGenericMatrix &mat, Scalar s, Scalar t) :
        n(mat.rows()),
        mat_H(n, n),
        ref_u(3, n - 1),
        computed(false)
    {
        compute(mat, s, t);
    }

    void compute(ConstGenericMatrix &mat, Scalar s, Scalar t)
    {
        if(mat.rows() != mat.cols())
            throw std::invalid_argument("DoubleShiftQR: matrix must be square");

        n = mat.rows();
        mat_H.resize(n, n);
        ref_u.resize(3, n - 1);

        mat_H = mat.template triangularView<Eigen::Upper>();
        mat_H.diagonal(-1) = mat.diagonal(-1);

        // Calculate the first reflector
        Scalar x = mat_H(0, 0) * (mat_H(0, 0) - s) + mat_H(0, 1) * mat_H(1, 0) + t;
        Scalar y = mat_H(1, 0) * (mat_H(0, 0) + mat_H(1, 1) - s);
        Scalar z = mat_H(2, 1) * mat_H(1, 0);
        compute_reflector(x, y, z, 0);
        // Apply the first reflector
        apply_PX(mat_H.template topRows<3>(), 0);
        apply_XP(mat_H.topLeftCorner(std::min(n, 4), 3), 0);

        // Transfrom mat_H to upper Hessenberg
        for(int i = 1; i < n - 2; i++)
        {
            compute_reflector(&mat_H(i, i - 1), i);
            // Apply the reflector to H
            apply_PX(mat_H.block(i, i - 1, 3, n - i + 1), i);
            apply_XP(mat_H.block(0, i, std::min(n, i + 4), 3), i);
        }

        // The last reflector
        compute_reflector(mat_H(n - 2, n - 3), mat_H(n - 1, n - 3), 0, n - 2);
        // Apply the reflector to H
        apply_PX(mat_H.template block<2, 3>(n - 2, n - 3), n - 2);
        apply_XP(mat_H.block(0, n - 2, n, 2), n - 2);

        computed = true;
    }

    Matrix matrix_QtHQ()
    {
        if(!computed)
            throw std::logic_error("DoubleShiftQR: need to call compute() first");

        return mat_H;
    }

    void apply_QtY(Vector &Y)
    {
        if(!computed)
            throw std::logic_error("DoubleShiftQR: need to call compute() first");
    }

    void apply_YQ(GenericMatrix Y)
    {
        if(!computed)
            throw std::logic_error("UpperHessenbergQR: need to call compute() first");
    }
};


#endif // DOUBLE_SHIFT_QR_H
