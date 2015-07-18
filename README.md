# ARPACK-Eigen

**ARPACK-Eigen** is a redesign of the [ARPACK](http://www.caam.rice.edu/software/ARPACK/)
software for large scale eigenvalue problems, built on top of
[Eigen](http://eigen.tuxfamily.org), an open source C++ linear algebra library.

**ARPACK-Eigen** is implemented as a header-only C++ library, whose only dependency,
**Eigen**, is also header-only. Hence **ARPACK-Eigen** can be easily embedded in
C++ projects that require solving large scale eigenvalue problems.

## Common Usage

**ARPACK-Eigen** is designed to calculate a specified number (`k`) of eigenvalues
of a large square matrix (`A`). Usually `k` is much less than the size of matrix
(`n`), so that only a few eigenvalues and eigenvectors are computed, which
in general is more efficient than calculating the whole spectral decomposition.

There are two major steps to use **ARPACK-Eigen** for calculating eigenvalues:

1. Define a class that could calculate the matrix-vector multiplication `A * x`
where `x` is any given real-valued vector. If the matrix `A` is already stored as a
matrix in **Eigen**, this step can be easily done by using the built-in classes
defined in **ARPACK-Eigen** which are simple wrappers of existing **Eigen** matrices.
2. Create an object of the eigen solver class, for example `SymEigsSolver` for
symmetric matrices, and `GenEigsSolver` for general matrices, set up the parameters
and options, and call member functions of this object to compute and retrieve the
eigenvalues and/or eigenvectors.

## Examples

Retrieving the largest (in magnitude) three eigenvalues and corresponding
eigenvectors of a real symmetric matrix.

```cpp
#include <Eigen/Core>
#include <iostream>
#include <SymEigsSolver.h>
#include <MatOp/DenseGenMatProd.h>

int main()
{
    srand(123);
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(10, 10);
    Eigen::MatrixXd mat = A.transpose() + A;
    int k = 3;
    int m = 6;

    DenseGenMatProd<double> op(mat);
    SymEigsSolver< double, LARGEST_MAGN, DenseGenMatProd<double> > eigs(&op, k, m);
    eigs.init();
    eigs.compute();

    Eigen::VectorXd evals = eigs.eigenvalues();
    Eigen::MatrixXd evecs = eigs.eigenvectors();

    std::cout << "Eigenvalues:\n" << evals << std::endl;
    std::cout << "Eigenvectors:\n" << evecs << std::endl;

    return 0;
}
```

## Advanced Usage

TODO

## License

**ARPACK-Eigen** is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/), the same license used by **Eigen**.
