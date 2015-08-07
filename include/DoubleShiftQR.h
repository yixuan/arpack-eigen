#ifndef DOUBLE_SHIFT_QR_H
#define DOUBLE_SHIFT_QR_H

#include <Eigen/Core>
#include <vector>
#include <stdexcept>

template <typename Scalar = double>
class DoubleShiftQR
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Matrix3X;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    typedef Eigen::Ref<Matrix> GenericMatrix;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    int n;              // Dimension of the matrix
    Matrix mat_H;       // A copy of the matrix to be factorized
    Scalar shift_s;     // Shift constant
    Scalar shift_t;     // Shift constant
    Matrix3X ref_u;     // Householder reflectors
    const Scalar prec;  // Approximately zero
    bool computed;      // Whether matrix has been factorized

    void compute_reflector(const Scalar &x1, const Scalar &x2, const Scalar &x3, int ind)
    {
        if(std::abs(x1) + std::abs(x2) + std::abs(x3) <= 3 * prec)
        {
            ref_u.col(ind).setZero();
            return;
        }
        // x1' = x1 - rho * ||x||
        // rho = -sign(x1)
        Scalar tmp = x2 * x2 + x3 * x3;
        Scalar x1_new = x1 - ((x1 < 0) - (x1 > 0)) * std::sqrt(x1 * x1 + tmp);
        Scalar x_norm = std::sqrt(x1_new * x1_new + tmp);
        ref_u(0, ind) = x1_new / x_norm;
        ref_u(1, ind) = x2 / x_norm;
        ref_u(2, ind) = x3 / x_norm;
    }

    void compute_reflector(const Scalar *x, int ind)
    {
        compute_reflector(x[0], x[1], x[2], ind);
    }

    void compute_reflectors_from_block(GenericMatrix X, int start_ind)
    {
        // For the block X, we can assume that ncol == nrow,
        // and all sub-diagonal elements are non-zero
        const int nrow = X.rows();
        // For block size == 1, there is no need to apply reflectors
        if(nrow == 1)
        {
            compute_reflector(0, 0, 0, start_ind);
            return;
        }

        // For block size == 2, do a Givens rotation on M = X * X - s * X + t * I
        if(nrow == 2) {
            Scalar x = X(0, 0) * (X(0, 0) - shift_s) + X(0, 1) * X(1, 0) + shift_t;
            Scalar y = X(1, 0) * (X(0, 0) + X(1, 1) - shift_s);
            compute_reflector(x, y, 0, start_ind);
            apply_PX(X.block(0, 0, 2, 2), start_ind);
            apply_XP(X.block(0, 0, 2, 2), start_ind);
            compute_reflector(0, 0, 0, start_ind + 1);
            return;
        }

        // For block size >=3, use the regular strategy
        Scalar x = X(0, 0) * (X(0, 0) - shift_s) + X(0, 1) * X(1, 0) + shift_t;
        Scalar y = X(1, 0) * (X(0, 0) + X(1, 1) - shift_s);
        Scalar z = X(2, 1) * X(1, 0);
        compute_reflector(x, y, z, start_ind);
        // Apply the first reflector
        apply_PX(X.template topRows<3>(), start_ind);
        apply_XP(X.topLeftCorner(std::min(nrow, 4), 3), start_ind);

        // Calculate the following reflectors
        // If entering this loop, nrow is at least 4.
        for(int i = 1; i < nrow - 2; i++)
        {
            compute_reflector(&X(i, i - 1), start_ind + i);
            // Apply the reflector to X
            apply_PX(X.block(i, i - 1, 3, nrow - i + 1), start_ind + i);
            apply_XP(X.block(0, i, std::min(nrow, i + 4), 3), start_ind + i);
        }

        // The last reflector
        compute_reflector(X(nrow - 2, nrow - 3), X(nrow - 1, nrow - 3), 0, start_ind + nrow - 2);
        // Apply the reflector to X
        apply_PX(X.template block<2, 3>(nrow - 2, nrow - 3), start_ind + nrow - 2);
        apply_XP(X.block(0, nrow - 2, nrow, 2), start_ind + nrow - 2);

        compute_reflector(0, 0, 0, start_ind + nrow - 1);
    }

    // P = I - 2 * u * u' = P'
    // PX = X - 2 * u * (u'X)
    void apply_PX(GenericMatrix X, int u_ind)
    {
        const Scalar sqrt_2 = std::sqrt(Scalar(2));

        Scalar u0 = sqrt_2 * ref_u(0, u_ind),
               u1 = sqrt_2 * ref_u(1, u_ind),
               u2 = sqrt_2 * ref_u(2, u_ind);

        if(std::abs(u0) + std::abs(u1) + std::abs(u2) <= 3 * sqrt_2 * prec)
            return;

        const int nrow = X.rows();
        const int ncol = X.cols();

        if(nrow == 2)
        {
            Scalar *xptr;
            for(int i = 0; i < ncol; i++)
            {
                xptr = &X(0, i);
                Scalar tmp = u0 * xptr[0] + u1 * xptr[1];
                xptr[0] -= tmp * u0;
                xptr[1] -= tmp * u1;
            }
        } else {
            Scalar *xptr;
            for(int i = 0; i < ncol; i++)
            {
                xptr = &X(0, i);
                Scalar tmp = u0 * xptr[0] + u1 * xptr[1] + u2 * xptr[2];
                xptr[0] -= tmp * u0;
                xptr[1] -= tmp * u1;
                xptr[2] -= tmp * u2;
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

        if(std::abs(u0) + std::abs(u1) + std::abs(u2) <= 3 * prec)
            return;

        // When the reflector only contains two elements, u2 has been set to zero
        bool u2_is_zero = (std::abs(u2) <= prec);
        Scalar dot2 = x[0] * u0 + x[1] * u1 + (u2_is_zero ? 0 : (x[2] * u2));
        dot2 *= 2;
        x[0] -= dot2 * u0;
        x[1] -= dot2 * u1;
        if(!u2_is_zero)
            x[2] -= dot2 * u2;
    }

    // XP = X - 2 * (X * u) * u'
    void apply_XP(GenericMatrix X, int u_ind)
    {
        const Scalar sqrt_2 = std::sqrt(Scalar(2));
        Scalar u0 = sqrt_2 * ref_u(0, u_ind),
               u1 = sqrt_2 * ref_u(1, u_ind),
               u2 = sqrt_2 * ref_u(2, u_ind);

        if(std::abs(u0) + std::abs(u1) + std::abs(u2) <= 3 * sqrt_2 * prec)
            return;

        const int nrow = X.rows();
        const int ncol = X.cols();
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
        prec(std::numeric_limits<Scalar>::epsilon()),
        computed(false)
    {}

    DoubleShiftQR(ConstGenericMatrix &mat, Scalar s, Scalar t) :
        n(mat.rows()),
        mat_H(n, n),
        shift_s(s),
        shift_t(t),
        ref_u(3, n),
        prec(std::numeric_limits<Scalar>::epsilon()),
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
        shift_s = s;
        shift_t = t;
        ref_u.resize(3, n);

        mat_H = mat.template triangularView<Eigen::Upper>();
        mat_H.diagonal(-1) = mat.diagonal(-1);

        Scalar prec2 = std::min(std::pow(prec, Scalar(2) / 3), n * prec);

        // Obtain the indices of zero elements in the subdiagonal,
        // so that H can be divided into several blocks
        std::vector<int> zero_ind;
        zero_ind.reserve(n - 1);
        zero_ind.push_back(0);
        for(int i = 1; i < n - 1; i++)
        {
            if(std::abs(mat_H(i, i - 1)) <= prec2)
            {
                mat_H(i, i - 1) = 0;
                zero_ind.push_back(i);
            }
        }
        zero_ind.push_back(n);

        for(std::vector<int>::size_type i = 0; i < zero_ind.size() - 1; i++)
        {
            int start = zero_ind[i];
            int end = zero_ind[i + 1] - 1;
            int block_size = end - start + 1;
            // Compute refelctors from each block X
            compute_reflectors_from_block(mat_H.block(start, start, block_size, block_size), start);
            // Apply reflectors to the block right to X
            if(end < n - 1 && block_size >= 2)
            {
                for(int j = start; j < end; j++)
                {
                    apply_PX(mat_H.block(j, end + 1, std::min(3, end - j + 1), n - 1 - end), j);
                }
            }
            // Apply reflectors to the block above X
            if(start > 0 && block_size >= 2)
            {
                for(int j = start; j < end; j++)
                {
                    apply_XP(mat_H.block(0, j, start, std::min(3, end - j + 1)), j);
                }
            }
        }

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
