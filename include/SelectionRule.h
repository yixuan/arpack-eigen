#ifndef SELECTION_RULE_H
#define SELECTION_RULE_H

#include <cmath>
#include <complex>
#include <utility>

enum SELECT_EIGENVALUE
{
    LARGEST_MAGN = 0,
    LARGEST_REAL,
    LARGEST_IMAG,
    LARGEST_ALGE,
    SMALLEST_MAGN,
    SMALLEST_REAL,
    SMALLEST_IMAG,
    SMALLEST_ALGE,
    BOTH_ENDS
};

// Default comparator: largest value come on the left
// This covers [float, double] x [LARGEST_REAL, LARGEST_ALGE]
//
// BOTH_ENDS will also be attributed to this case, and we need
// to move those smallest values to the proper locations.
// This is done in SymEigsSolver.h => retrieve_ritzpair()
template <typename Scalar, int SelectionRule>
class EigenvalueComparator
{
public:
    typedef std::pair<Scalar, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first > v2.first;
    }
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
template <>
class EigenvalueComparator<std::complex<double>, LARGEST_REAL>
{
public:
    typedef std::pair<std::complex<double>, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first.real() > v2.first.real();
    }
};

// Specialization for LARGEST_IMAG
// This covers [complex] x [LARGEST_IMAG]
template <>
class EigenvalueComparator<std::complex<double>, LARGEST_IMAG>
{
public:
    typedef std::pair<std::complex<double>, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first.imag() > v2.first.imag();
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
template <>
class EigenvalueComparator<std::complex<double>, SMALLEST_REAL>
{
public:
    typedef std::pair<std::complex<double>, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first.real() <= v2.first.real();
    }
};

// Specialization for SMALLEST_IMAG
// This covers [complex] x [SMALLEST_IMAG]
template <>
class EigenvalueComparator<std::complex<double>, SMALLEST_IMAG>
{
public:
    typedef std::pair<std::complex<double>, int> SortPair;

    bool operator() (SortPair v1, SortPair v2)
    {
        return v1.first.imag() <= v2.first.imag();
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


#endif // SELECTION_RULE_H
