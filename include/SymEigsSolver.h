#ifndef SYMEIGSSOLVER_H
#define SYMEIGSSOLVER_H

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>
#include "MatOp.h"

template <typename Scalar>
class SymEigsSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;
    typedef Eigen::Array<bool, Eigen::Dynamic, 1> BoolArray;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::SelfAdjointEigenSolver<Matrix> EigenSolver;
    typedef Eigen::HouseholderQR<Matrix> QRdecomp;
    typedef Eigen::HouseholderSequence<Matrix, Vector> QRQ;
    typedef std::pair<Scalar, int> SortPair;

    MatOp<Scalar> *op;
    int dim_n;           // dimension of A
    int nev;             // number of eigenvalues requested
    int ncv;             // "m"
    int fac_step;        // how many steps in the Arnoldi factorization
    Matrix fac_V;
    Matrix fac_H;
    Vector fac_f;

    Vector ritz_val;
    Matrix ritz_vec;
    BoolArray ritz_conv;

    // Arnoldi factorization
    void factorize_from(int from_k, int to_m, const Vector &fk)
    {
        if(to_m <= from_k) return;

        fac_f = fk;

        Vector v(dim_n);
        Scalar beta = 0.0;
        for(int i = from_k; i <= to_m - 1; i++)
        {
            beta = fac_f.norm();
            v.noalias() = fac_f / beta;
            fac_V.col(i) = v; // The (i+1)-th column
            fac_H.block(i, 0, 1, i).setZero();
            fac_H(i, i - 1) = beta;

            Vector w(dim_n);
            op->prod(v.data(), w.data());
            Vector h = fac_V.leftCols(i + 1).adjoint() * w;
            fac_f = w - fac_V.leftCols(i + 1) * h;
            fac_H.block(0, i, i + 1, 1) = h;
        }

        fac_step = to_m;
    }

    void factorize()
    {
        factorize_from(1, ncv, fac_f);
        retrieve_ritzpair();
    }

    static bool compare_eigenvalue(SortPair p1, SortPair p2)
    {
        return p1.first > p2.first;
    }

    void retrieve_ritzpair()
    {
        EigenSolver eig(fac_H);
        Vector evals = eig.eigenvalues();
        Matrix evecs = eig.eigenvectors();

        std::vector<SortPair> pairs(ncv);
        for(int i = 0; i < ncv; i++)
        {
            pairs[i].first = evals[i];
            pairs[i].second = i;
        }

        std::sort(pairs.begin(), pairs.end(), compare_eigenvalue);

        for(int i = 0; i < ncv; i++)
        {
            ritz_val[i] = pairs[i].first;
        }
        for(int i = 0; i < nev; i++)
        {
            ritz_vec.col(i) = evecs.col(pairs[i].second);
        }
    }

    void restart(int k)
    {
        if(fac_step != ncv)
            return;
        if(k >= ncv)
            return;

        QRdecomp qr;
        Vector em(ncv);
        em.setZero();
        em[ncv - 1] = 1;

        for(int i = k; i < ncv; i++)
        {
            qr.compute(fac_H - ritz_val[i] * Matrix::Identity(ncv, ncv));
            QRQ Q = qr.householderQ();
            // V -> VQ
            fac_V.applyOnTheRight(Q);
            // H -> Q'HQ
            fac_H.applyOnTheRight(Q);
            fac_H.applyOnTheLeft(Q.adjoint());
            em.applyOnTheLeft(Q.adjoint());
        }

        Vector fk = fac_f * em[k - 1];
        factorize_from(k, ncv, fk);
        retrieve_ritzpair();
    }

    bool converged(Scalar tol)
    {
        Scalar prec = std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(2.0 / 3));
        Array bound = tol * ritz_val.head(nev).array().abs().max(prec);
        Array resid =  ritz_vec.bottomRows(1).transpose().array().abs() * fac_f.norm();
        ritz_conv = (resid < bound);

        return ritz_conv.all();
    }

    void init()
    {
        fac_V.setZero();
        fac_H.setZero();
        fac_f.setZero();
        ritz_val.setZero();
        ritz_vec.setZero();
        ritz_conv.setZero();

        Vector v = Vector::Random(dim_n);
        v.normalize();
        Vector w(dim_n);
        op->prod(v.data(), w.data());

        fac_H(0, 0) = v.dot(w);
        fac_f = w - v * fac_H(0, 0);
        fac_V.col(0) = v;
    }

public:
    SymEigsSolver(MatOp<Scalar> *op_, int nev_, int ncv_) :
        op(op_),
        dim_n(op->rows()),
        nev(nev_),
        ncv(ncv_),
        fac_step(1),
        fac_V(dim_n, ncv),
        fac_H(ncv, ncv),
        fac_f(dim_n),
        ritz_val(ncv),
        ritz_vec(ncv, nev),
        ritz_conv(nev)
    {
        init();
    }

    int compute(int maxit = 1000, Scalar tol = 1e-10)
    {
        factorize();

        int i = 0;
        for(i = 0; i < maxit; i++)
        {
            if(converged(tol))
                break;

            restart(nev);
        }

        return i + 1;
    }

    Vector eigenvalues()
    {
        int nconv = ritz_conv.cast<int>().sum();
        Vector res(nconv);

        if(!nconv)
            return res;

        int j = 0;
        for(int i = 0; i < nev; i++)
        {
            if(ritz_conv[i])
            {
                res[j] = ritz_val[i];
                j++;
            }
        }

        return res;
    }

    Matrix eigenvectors()
    {
        int nconv = ritz_conv.cast<int>().sum();
        Matrix res(dim_n, nconv);

        if(!nconv)
            return res;

        Matrix ritz_vec_conv(ncv, nconv);
        int j = 0;
        for(int i = 0; i < nev; i++)
        {
            if(ritz_conv[i])
            {
                ritz_vec_conv.col(j) = ritz_vec.col(i);
                j++;
            }
        }

        return fac_V * ritz_vec_conv;
    }
};

#endif // SYMEIGSSOLVER_H
