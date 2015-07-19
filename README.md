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

Moreover, in the basic setting the underlying algorithm of **ARPACK-Eigen**
(and also **ARPACK**) only requires matrix-vector multiplication `A` to calculate
eigenvalues. Therefore, if `A * x` can be computed efficiently, which is the case
when `A` is sparse, **ARPACK-Eigen** will be very powerful for large scale eigenvalue problems.

There are two major steps to use the **ARPACK-Eigen** library:

1. Define a class that could calculate the matrix-vector multiplication `A * x`
where `x` is any given real-valued vector. If the matrix `A` is already stored as a
matrix in **Eigen**, this step can be easily done by using the built-in classes
defined in **ARPACK-Eigen** which are simple wrappers of existing **Eigen** matrices.
2. Create an object of the eigen solver class, for example `SymEigsSolver` for
symmetric matrices, and `GenEigsSolver` for general matrices, set up the parameters
and options, and call member functions of this object to compute and retrieve the
eigenvalues and/or eigenvectors.

This can be better exaplained by the example below, which calculates the largest
(in magnitude, or equivalently, absolute value) three eigenvalues and corresponding
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

    DenseGenMatProd<double> op(mat);                          // [1]
    SymEigsSolver< double, LARGEST_MAGN,
                   DenseGenMatProd<double> > eigs(&op, k, m); // [2]
    eigs.init();                                              // [3]
    eigs.compute();                                           // [4]

    Eigen::VectorXd evals = eigs.eigenvalues();               // [5]
    Eigen::MatrixXd evecs = eigs.eigenvectors();              // [6]

    std::cout << "Eigenvalues:\n" << evals << std::endl;
    std::cout << "Eigenvectors:\n" << evecs << std::endl;

    return 0;
}
```

In this example we calculate the eigenvalues of a matrix that is stored as a
**Eigen** matrix type, and Line [1] wraps this matrix by the
`DenseGenMatprod<double>` class that has already been defined in **ARPACK-Eigen**.

Line [2] constructs an instance of the eigen solver class `SymEigsSolver`, with
the template parameter `LARGEST_MAGN` indicating that we need eigenvalues with
largest magnitude.

Line [3] to [6] are API function calls that do the actual computation.

## Advanced Features

TODO

## License

**ARPACK-Eigen** is an open source project licensed under
[MPL2](https://www.mozilla.org/MPL/2.0/), the same license used by **Eigen**.
