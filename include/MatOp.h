#ifndef MATOP_H
#define MATOP_H

template <typename Scalar>
class MatOp
{
private:
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
class MatOpWithTransProd: public virtual MatOp<Scalar>
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
class MatOpWithRealShiftSolve: public virtual MatOp<Scalar>
{
public:
    // Constructor
    MatOpWithRealShiftSolve(int m_, int n_) :
        MatOp<Scalar>(m_, n_)
    {}
    // Destructor
    virtual ~MatOpWithRealShiftSolve() {}

    // setting sigma
    virtual void set_real_shift(Scalar sigma) {}
    // y_out = inv(A - sigma * I) * x_in
    virtual void shift_solve(Scalar *x_in, Scalar *y_out) = 0;
};

template <typename Scalar>
class MatOpWithComplexShiftSolve: public MatOpWithRealShiftSolve<Scalar>
{
protected:
    bool sigma_is_real;
    // shift solve for real sigma
    virtual void real_shift_solve(Scalar *x_in, Scalar *y_out)
    {
        complex_shift_solve(x_in, y_out);
    }
    // shift solve for complex sigma
    virtual void complex_shift_solve(Scalar *x_in, Scalar *y_out) = 0;
public:
    // Constructor
    MatOpWithComplexShiftSolve(int m_, int n_) :
        MatOpWithRealShiftSolve<Scalar>(m_, n_),
        sigma_is_real(false)
    {}
    // Destructor
    virtual ~MatOpWithComplexShiftSolve() {}

    // setting real shift
    virtual void set_real_shift(Scalar sigma)
    {
        set_complex_shift(sigma, Scalar(0));
        sigma_is_real = true;
    }
    // setting complex shift
    virtual void set_complex_shift(Scalar sigmar, Scalar sigmai) {}
    // y_out = inv(A - sigma * I) * x_in
    virtual void shift_solve(Scalar *x_in, Scalar *y_out)
    {
        if(sigma_is_real)
            real_shift_solve(x_in, y_out);
        else
            complex_shift_solve(x_in, y_out);
    }
};


#endif // MATOP_H
