#ifndef MATOP_H
#define MATOP_H

#include <Eigen/Dense>

template <typename Scalar>
class MatOp
{
protected:
    // Dimension of the matrix
    // m rows and n columns
    // In eigenvalue problems, they are assumed to be equal,
    // and only n is used.
    int m;
    int n;

public:
    // Constructor
    MatOp(int m_, int n_) :
        m(m_), n(n_)
    {}
    // Destructor
    virtual ~MatOp() {}

    // y_out = A * x_in
    virtual void prod(Scalar *x_in, Scalar *y_out) = 0;

    int rows() { return m; }
    int cols() { return n; }
};

template <typename Scalar>
class MatOpWithTransProd: public MatOp<Scalar>
{
public:
    // Constructor
    MatOpWithTransProd(int m_, int n_) :
        MatOp<Scalar>(m_, n_)
    {}
    // Destructor
    virtual ~MatOpWithTransProd() {}

    // y_out = A' * x_in
    virtual void trans_prod(Scalar *x_in, Scalar *y_out) = 0;
};

template <typename Scalar>
class MatOpWithShiftSolve: public MatOpWithTransProd<Scalar>
{
public:
    // Constructor
    MatOpWithShiftSolve(int m_, int n_) :
        MatOpWithTransProd<Scalar>(m_, n_)
    {}
    // Destructor
    virtual ~MatOpWithShiftSolve() {}

    // y_out = A' * x_in
    virtual void trans_prod(Scalar *x_in, Scalar *y_out) {}
    // setting sigmar and sigmai
    virtual void set_shift(Scalar sigmar, Scalar sigmai) {}
    // y_out = inv(A - sigma * I) * x_in
    virtual void shift_solve(Scalar *x_in, Scalar *y_out) = 0;
};



template <typename Scalar>
class DenseMatOp: public MatOpWithShiftSolve<Scalar>
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;

    MapMat mat;
    MapVec vec_x;
    MapVec vec_y;
public:
    DenseMatOp(const Matrix &mat_) :
        MatOpWithShiftSolve<Scalar>(mat_.rows(), mat_.cols()),
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        vec_x(NULL, 1),
        vec_y(NULL, 1)
    {}

    virtual ~DenseMatOp() {}

    // y_out = A * x_in
    virtual void prod(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, MatOp<Scalar>::n);
        new (&vec_y) MapVec(y_out, MatOp<Scalar>::m);

        vec_y.noalias() = mat * vec_x;
    }

    // y_out = A' * x_in
    virtual void trans_prod(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, MatOp<Scalar>::m);
        new (&vec_y) MapVec(y_out, MatOp<Scalar>::n);

        vec_y.noalias() = mat.transpose() * vec_x;
    }

    // setting sigmar and sigmai
    virtual void set_shift(Scalar sigmar, Scalar sigmai)
    {

    }
    // y_out = inv(A - sigma * I) * x_in
    virtual void shift_solve(Scalar *x_in, Scalar *y_out)
    {

    }
};




#endif // MATOP_H
