# ARPACK-Eigen

> A redesign of the [ARPACK](http://www.caam.rice.edu/software/ARPACK/)
software for large scale eigenvalue problems, built on top of
[Eigen](http://eigen.tuxfamily.org), an open source C++ linear algebra library.

**ARPACK-Eigen** is a header-only library implemented using the modern C++
language, whose only dependency, **Eigen**, is also a header-only library.
Hence **ARPACK-Eigen** is easy to be embedded in C++ projects that require
solving large scale eigenvalue problems.

## Common Usage

TODO

## Examples

Retrieving the largest (in magnitude) three eigenvalues and corresponding
eigenvectors of a real symmetric matrix.

```cpp
#include <Eigen/Dense>
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
    SymEigsSolver<double, LARGEST_MAGN, DenseGenMatProd<double>> eigs(&op, k, m);
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
