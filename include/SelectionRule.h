#ifndef SELECTION_RULE_H
#define SELECTION_RULE_H

#include <cmath>
#include <complex>
#include <utility>

///
/// \file SelectionRule.h
///
/// This file defines enumeration types for the selection rule of eigenvalues.
///

///
/// The enumeration of selection rules of desired eigenvalues.
///
enum SELECT_EIGENVALUE
{

    LARGEST_MAGN = 0,  ///< Select eigenvalues with largest magnitude. Magnitude
                       ///< means the absolute value for real numbers and norm for
                       ///< complex numbers. Applies to both symmetric and general
                       ///< eigen solvers.

    LARGEST_REAL,      ///< Select eigenvalues with largest real part. Only for general eigen solvers.

    LARGEST_IMAG,      ///< Select eigenvalues with largest imaginary part (in magnitude). Only for general eigen solvers.

    LARGEST_ALGE,      ///< Select eigenvalues with largest algebraic value, considering
                       ///< any negative sign. Only for symmetric eigen solvers.

    SMALLEST_MAGN,     ///< Select eigenvalues with smallest magnitude. Applies to both symmetric and general
                       ///< eigen solvers.

    SMALLEST_REAL,     ///< Select eigenvalues with smallest real part. Only for general eigen solvers.

    SMALLEST_IMAG,     ///< Select eigenvalues with smallest imaginary part (in magnitude). Only for general eigen solvers.

    SMALLEST_ALGE,     ///< Select eigenvalues with smallest algebraic value. Only for symmetric eigen solvers.

    BOTH_ENDS          ///< Select eigenvalues half from each end of the spectrum. When
                       ///< `nev` is odd, compute more from the high end. Only for symmetric eigen solvers.
};

///
/// The enumeration of selection rules of desired eigenvalues. Alias for `SELECT_EIGENVALUE`.
///
enum SELECT_EIGENVALUE_ALIAS
{
    WHICH_LM = 0,  ///< Alias for `LARGEST_MAGN`
    WHICH_LR,      ///< Alias for `LARGEST_REAL`
    WHICH_LI,      ///< Alias for `LARGEST_IMAG`
    WHICH_LA,      ///< Alias for `LARGEST_ALGE`
    WHICH_SM,      ///< Alias for `SMALLEST_MAGN`
    WHICH_SR,      ///< Alias for `SMALLEST_REAL`
    WHICH_SI,      ///< Alias for `SMALLEST_IMAG`
    WHICH_SA,      ///< Alias for `SMALLEST_ALGE`
    WHICH_BE       ///< Alias for `BOTH_ENDS`
};

/// \cond

// Default comparator: an empty class without
// operator() definition, so it won't compile
// when operator() is called on this class
template <typename Scalar, int SelectionRule>
class EigenvalueComparator
{
public:
    typedef std::pair<Scalar, int> SortPair;
};

// Specialization for LARGEST_MAGN
// This covers [float, double, complex] x [LARGEST_MAGN]
template <typename Scalar>
class EigenvalueComparator<Scalar, LARGEST_MAGN>
{
public:
    typedef std::pair<Scalar, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return std::abs(v1.first) > std::abs(v2.first);
    }
};

// Specialization for LARGEST_REAL
// This covers [complex] x [LARGEST_REAL]
template <typename RealType>
class EigenvalueComparator<std::complex<RealType>, LARGEST_REAL>
{
public:
    typedef std::pair<std::complex<RealType>, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first.real() > v2.first.real();
    }
};

// Specialization for LARGEST_IMAG
// This covers [complex] x [LARGEST_IMAG]
template <typename RealType>
class EigenvalueComparator<std::complex<RealType>, LARGEST_IMAG>
{
public:
    typedef std::pair<std::complex<RealType>, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return std::abs(v1.first.imag()) > std::abs(v2.first.imag());
    }
};

// Specialization for LARGEST_ALGE
// This covers [float, double] x [LARGEST_ALGE]
template <typename Scalar>
class EigenvalueComparator<Scalar, LARGEST_ALGE>
{
public:
    typedef std::pair<Scalar, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first > v2.first;
    }
};

// Here BOTH_ENDS is the same as LARGEST_ALGE, but
// we need some additional steps, which are done in
// SymEigsSolver.h => retrieve_ritzpair().
// There we move the smallest values to the proper locations.
template <typename Scalar>
class EigenvalueComparator<Scalar, BOTH_ENDS>
{
public:
    typedef std::pair<Scalar, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first > v2.first;
    }
};

// Specialization for SMALLEST_MAGN
// This covers [float, double, complex] x [SMALLEST_MAGN]
template <typename Scalar>
class EigenvalueComparator<Scalar, SMALLEST_MAGN>
{
public:
    typedef std::pair<Scalar, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return std::abs(v1.first) <= std::abs(v2.first);
    }
};

// Specialization for SMALLEST_REAL
// This covers [complex] x [SMALLEST_REAL]
template <typename RealType>
class EigenvalueComparator<std::complex<RealType>, SMALLEST_REAL>
{
public:
    typedef std::pair<std::complex<RealType>, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first.real() <= v2.first.real();
    }
};

// Specialization for SMALLEST_IMAG
// This covers [complex] x [SMALLEST_IMAG]
template <typename RealType>
class EigenvalueComparator<std::complex<RealType>, SMALLEST_IMAG>
{
public:
    typedef std::pair<std::complex<RealType>, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return std::abs(v1.first.imag()) <= std::abs(v2.first.imag());
    }
};

// Specialization for SMALLEST_ALGE
// This covers [float, double] x [SMALLEST_ALGE]
template <typename Scalar>
class EigenvalueComparator<Scalar, SMALLEST_ALGE>
{
public:
    typedef std::pair<Scalar, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first <= v2.first;
    }
};

/// \endcond

#endif // SELECTION_RULE_H
