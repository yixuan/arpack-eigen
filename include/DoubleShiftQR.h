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
    // Approximately zero
    const Scalar prec;
    bool new_start;
    bool computed;

    void compute_reflector(const Scalar &x1, const Scalar &x2, const Scalar &x3, int ind)
    {
        Scalar tmp = x2 * x2 + x3 * x3;
        // x1' = x1 - rho * ||x||
        // rho = -sign(x1)
        Scalar x1_new = x1 - ((x1 < 0) - (x1 > 0)) * std::sqrt(x1 * x1 + tmp);
        Scalar x_norm = std::sqrt(x1_new * x1_new + tmp);
        if(x_norm <= prec)
        {
            ref_u(0, ind) = 0;
            ref_u(1, ind) = 0;
            ref_u(2, ind) = 0;
        } else {
            ref_u(0, ind) = x1_new / x_norm;
            ref_u(1, ind) = x2 / x_norm;
            ref_u(2, ind) = x3 / x_norm;
        }
    }

    void compute_reflector(const Scalar *x, int ind)
    {
        compute_reflector(x[0], x[1], x[2], ind);
    }

    // P = I - 2 * u * u' = P'
    // PX = X - 2 * u * (u'X)
    void apply_PX(GenericMatrix X, int u_ind)
    {
        const int nrow = X.rows();
        const int ncol = X.cols();
        const Scalar sqrt_2 = std::sqrt(Scalar(2));

        Scalar u0 = sqrt_2 * ref_u(0, u_ind),
               u1 = sqrt_2 * ref_u(1, u_ind),
               u2 = sqrt_2 * ref_u(2, u_ind);

        if(nrow == 2)
        {
            for(int i = 0; i < ncol; i++)
            {
                Scalar tmp = u0 * X(0, i) + u1 * X(1, i);
                X(0, i) -= tmp * u0;
                X(1, i) -= tmp * u1;
            }
        } else {
            for(int i = 0; i < ncol; i++)
            {
                Scalar tmp = u0 * X(0, i) + u1 * X(1, i) + u2 * X(2, i);
                X(0, i) -= tmp * u0;
                X(1, i) -= tmp * u1;
                X(2, i) -= tmp * u2;
            }
        }
    }

    // x is a pointer to a vector
    // Px = x - 2 * dot(x, u) * u
    void apply_PX(Scalar *x, int u_ind)
    {
        Scalar u0 = ref_u(0, u_ind),
               u1 = ref_u(1, u_ind),
               u2 = ref_u(2, u_ind);
        Scalar dot2 = x[0] * u0 + x[1] * u1 + ((u_ind == n - 2) ? 0 : (x[2] * u2));
        dot2 *= 2;
        x[0] -= dot2 * u0;
        x[1] -= dot2 * u1;
        if(u_ind < n - 2)
            x[2] -= dot2 * u2;
    }

    // XP = X - 2 * (X * u) * u'
    void apply_XP(GenericMatrix X, int u_ind)
    {
        const int nrow = X.rows();
        const int ncol = X.cols();
        const Scalar sqrt_2 = std::sqrt(Scalar(2));

        Scalar u0 = sqrt_2 * ref_u(0, u_ind),
               u1 = sqrt_2 * ref_u(1, u_ind),
               u2 = sqrt_2 * ref_u(2, u_ind);
        Scalar *X0 = &X(0, 0), *X1 = &X(0, 1);

        if(ncol == 2)
        {
            for(int i = 0; i < nrow; i++)
            {
                Scalar tmp = u0 * X0[i] + u1 * X1[i];
                X0[i] -= tmp * u0;
                X1[i] -= tmp * u1;
            }
        } else {
            Scalar *X2 = &X(0, 2);
            for(int i = 0; i < nrow; i++)
            {
                Scalar tmp = u0 * X0[i] + u1 * X1[i] + u2 * X2[i];
                X0[i] -= tmp * u0;
                X1[i] -= tmp * u1;
                X2[i] -= tmp * u2;
            }
        }
    }

public:
    DoubleShiftQR() :
        n(0),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(0.9))),
        new_start(false),
        computed(false)
    {}

    DoubleShiftQR(ConstGenericMatrix &mat, Scalar s, Scalar t) :
        n(mat.rows()),
        mat_H(n, n),
        ref_u(3, n - 1),
        prec(std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(0.9))),
        new_start(false),
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

        Scalar x, y, z;

        // If the subdiagonal element is zero, we can skip the first column
        // and have a new start in next iteration
        if(std::abs(mat_H(1, 0)) <= prec)
        {
            compute_reflector(0, 0, 0, 0);
            new_start = true;
        } else {
            // Calculate the first reflector
            x = mat_H(0, 0) * (mat_H(0, 0) - s) + mat_H(0, 1) * mat_H(1, 0) + t;
            y = mat_H(1, 0) * (mat_H(0, 0) + mat_H(1, 1) - s);
            z = mat_H(2, 1) * mat_H(1, 0);
            compute_reflector(x, y, z, 0);
            // Apply the first reflector
            apply_PX(mat_H.template topRows<3>(), 0);
            apply_XP(mat_H.topLeftCorner(std::min(n, 4), 3), 0);
        }

        // Calculate the following reflectors
        for(int i = 1; i < n - 2; i++)
        {
            // If entering this loop, n is at least 4.

            // First check: whether the subdiagonal element is zero
            if(std::abs(mat_H(i + 1, i)) <= prec)
            {
                compute_reflector(0, 0, 0, i);
                new_start = true;
            } else if(new_start) {  // Second check: whether this is a new start
                x = mat_H(i, i) * (mat_H(i, i) - s) + mat_H(i, i + 1) * mat_H(i + 1, i) + t;
                y = mat_H(i + 1, i) * (mat_H(i, i) + mat_H(i + 1, i + 1) - s);
                z = mat_H(i + 2, i + 1) * mat_H(i + 1, i);
                compute_reflector(x, y, z, i);
                // Apply the reflector to H
                apply_PX(mat_H.block(i, i, 3, n - i), i);
                apply_XP(mat_H.block(0, i, std::min(n, i + 4), 3), i);

                new_start = false;
            } else {
                compute_reflector(&mat_H(i, i - 1), i);
                // Apply the reflector to H
                apply_PX(mat_H.block(i, i - 1, 3, n - i + 1), i);
                apply_XP(mat_H.block(0, i, std::min(n, i + 4), 3), i);
            }
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

    // Q = P0 * P1 * ...
    // Q'y = P_{n-2} * ... * P1 * P0 * y
    void apply_QtY(Vector &y)
    {
        if(!computed)
            throw std::logic_error("DoubleShiftQR: need to call compute() first");

        Scalar *y_ptr = y.data();
        for(int i = 0; i < n - 1; i++, y_ptr++)
        {
            apply_PX(y_ptr, i);
        }
    }

    // Q = P0 * P1 * ...
    // YQ = Y * P0 * P1 * ...
    void apply_YQ(GenericMatrix Y)
    {
        if(!computed)
            throw std::logic_error("DoubleShiftQR: need to call compute() first");

        int nrow = Y.rows();
        for(int i = 0; i < n - 2; i++)
        {
            apply_XP(Y.block(0, i, nrow, 3), i);
        }
        apply_XP(Y.block(0, n - 2, nrow, 2), n - 2);
    }
};


#endif // DOUBLE_SHIFT_QR_H
