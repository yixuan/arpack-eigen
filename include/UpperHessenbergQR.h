#ifndef UpperHessenbergQR_H
#define UpperHessenbergQR_H

#include <Eigen/Dense>

// QR decomposition of an upper Hessenberg matrix
template <typename Scalar = double>
class UpperHessenbergQR
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, 2, 2> Matrix22;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

protected:
    int n;
    Matrix mat_T;
    // Gi = [ cos[i]  sin[i]]
    //      [-sin[i]  cos[i]]
    // Q = G1 * G2 * ... * G_{n-1}
    Array rot_cos;
    Array rot_sin;
public:
    UpperHessenbergQR() :
        n(0)
    {}

    UpperHessenbergQR(int n_) :
        n(n_),
        mat_T(n, n),
        rot_cos(n - 1),
        rot_sin(n - 1)
    {}

    UpperHessenbergQR(const Matrix &mat) :
        n(mat.rows()),
        mat_T(n, n),
        rot_cos(n - 1),
        rot_sin(n - 1)
    {
        compute(mat);
    }

    virtual void compute(const Matrix &mat)
    {
        n = mat.rows();
        mat_T.resize(n, n);
        rot_cos.resize(n - 1);
        rot_sin.resize(n - 1);

        mat_T = mat;

        Scalar xi, xj, r, c, s;
        for(int i = 0; i < n - 2; i++)
        {
            xi = mat_T(i, i);
            xj = mat_T(i + 1, i);
            r = std::sqrt(xi * xi + xj * xj);
            rot_cos[i] = c = xi / r;
            rot_sin[i] = s = -xj / r;
            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:n] = G' * T[i:(i + 1), i:n]
            // Since here we only want to obtain the cos and sin sequence,
            // we only update T[i + 1, (i + 1):n]
            mat_T.block(i + 1, i + 1, 1, n - i - 1) *= c;
            mat_T.block(i + 1, i + 1, 1, n - i - 1) += s * mat_T.block(i, i + 1, 1, n - i - 1);
            // Matrix Gt;
            // Gt << c << -s << arma::endr << s << c << arma::endr;
            // mat_T.rows(i, i + 1) = Gt * mat_T.rows(i, i + 1);
        }
        // For i = n - 2
        xi = mat_T(n - 2, n - 2);
        xj = mat_T(n - 1, n - 2);
        r = std::sqrt(xi * xi + xj * xj);
        rot_cos[n - 2] = xi / r;
        rot_sin[n - 2] = -xj / r;
    }

    // Y -> QY = G1 * G2 * ... * Y
    virtual void applyQY(Vector &Y)
    {
        Scalar c, s, Yi, Yi1;
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[i:(i + 1)] = Gi * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            c = rot_cos[i];
            s = rot_sin[i];
            Yi = Y[i];
            Yi1 = Y[i + 1];
            Y[i] = c * Yi + s * Yi1;
            Y[i + 1] = -s * Yi + c * Yi1;
        }
    }

    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    virtual void applyQtY(Vector &Y)
    {
        Scalar c, s, Yi, Yi1;
        for(int i = 0; i < n - 1; i++)
        {
            // Y[i:(i + 1)] = Gi' * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            c = rot_cos[i];
            s = rot_sin[i];
            Yi = Y[i];
            Yi1 = Y[i + 1];
            Y[i] = c * Yi - s * Yi1;
            Y[i + 1] = s * Yi + c * Yi1;
        }
    }

    // Y -> QY = G1 * G2 * ... * Y
    virtual void applyQY(Matrix &Y)
    {
        Matrix22 Gi;
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[i:(i + 1), ] = Gi * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Gi << rot_cos[i], rot_sin[i],
                 -rot_sin[i], rot_cos[i];

            Y.block(i, 0, 2, Y.cols()) = Gi * Y.block(i, 0, 2, Y.cols());
        }
    }

    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    virtual void applyQtY(Matrix &Y)
    {
        Matrix22 Git;
        for(int i = 0; i < n - 1; i++)
        {
            // Y[i:(i + 1), ] = Gi' * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Git << rot_cos[i], -rot_sin[i],
                   rot_sin[i],  rot_cos[i];

            Y.block(i, 0, 2, Y.cols()) = Git * Y.block(i, 0, 2, Y.cols());
        }
    }

    // Y -> YQ = Y * G1 * G2 * ...
    virtual void applyYQ(Matrix &Y)
    {
        Matrix22 Gi;
        for(int i = 0; i < n - 1; i++)
        {
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Gi << rot_cos[i], rot_sin[i],
                 -rot_sin[i], rot_cos[i];

            Y.block(0, i, Y.rows(), 2) = Y.block(0, i, Y.rows(), 2) * Gi;
        }
    }

    // Y -> YQ' = Y * G_{n-1}' * ... * G2' * G1'
    virtual void applyYQt(Matrix &Y)
    {
        Matrix22 Git;
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi'
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Git << rot_cos[i], -rot_sin[i],
                   rot_sin[i],  rot_cos[i];

            Y.block(0, i, Y.rows(), 2) = Y.block(0, i, Y.rows(), 2) * Git;
        }
    }
};



// QR decomposition of a tridiagonal matrix as a special case of
// upper Hessenberg matrix
template <typename Scalar = double>
class TridiagQR: public UpperHessenbergQR<Scalar>
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
    TridiagQR() :
        UpperHessenbergQR<Scalar>()
    {}

    TridiagQR(int n_) :
        UpperHessenbergQR<Scalar>(n_)
    {}

    TridiagQR(const Matrix &mat) :
        UpperHessenbergQR<Scalar>(mat.rows())
    {
        this->compute(mat);
    }

    virtual void compute(const Matrix &mat)
    {
        this->n = mat.rows();
        this->mat_T.resize(this->n, this->n);
        this->rot_cos.resize(this->n - 1);
        this->rot_sin.resize(this->n - 1);

        this->mat_T.setZero();
        this->mat_T.diagonal() = mat.diagonal();
        this->mat_T.diagonal(1) = mat.diagonal(-1);
        this->mat_T.diagonal(-1) = mat.diagonal(-1);

        Scalar xi, xj, r, c, s;
        for(int i = 0; i < this->n - 2; i++)
        {
            xi = this->mat_T(i, i);
            xj = this->mat_T(i + 1, i);
            r = std::sqrt(xi * xi + xj * xj);
            this->rot_cos[i] = c = xi / r;
            this->rot_sin[i] = s = -xj / r;
            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:(i + 2)] = G' * T[i:(i + 1), i:(i + 2)]
            // Since here we only want to obtain the cos and sin sequence,
            // we only update T[i + 1, (i + 1):(i + 2)]
            this->mat_T(i + 1, i + 1) = s * this->mat_T(i, i + 1) + c * this->mat_T(i + 1, i + 1);
            this->mat_T(i + 1, i + 2) *= c;

        }
        // For i = n - 2
        xi = this->mat_T(this->n - 2, this->n - 2);
        xj = this->mat_T(this->n - 1, this->n - 2);
        r = std::sqrt(xi * xi + xj * xj);
        this->rot_cos[this->n - 2] = xi / r;
        this->rot_sin[this->n - 2] = -xj / r;
    }
};



#endif // UpperHessenbergQR_H
