#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "util.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <random>
#include <utility>
#include <vector>

class Matrix {
public:
    Matrix( int row, int col ) : m_row( row ), m_col( col ) {
        data.resize( row * col );
    }
    int row() const { return m_row; }
    int col() const { return m_col; }
    double &operator()( int i, int j ) {
        return data[ i * m_col + j ];
    }
    double operator()( int i, int j ) const {
        return data[ i * m_col + j ];
    }
    Matrix operator+( const Matrix &rhs ) const {
        assert( m_row == rhs.m_row && m_col == rhs.m_col && "Matrix size mismatch" );
        Matrix result( m_row, m_col );
        result.data = data;
        std::transform( result.data.begin(), result.data.end(), rhs.data.begin(), result.data.begin(), std::plus<>() );
        return result;
    }
    Matrix operator-( const Matrix &rhs ) const {
        assert( m_row == rhs.m_row && m_col == rhs.m_col && "Matrix size mismatch" );
        Matrix result( m_row, m_col );
        result.data = data;
        std::transform( result.data.begin(), result.data.end(), rhs.data.begin(), result.data.begin(), std::minus<>() );
        return result;
    }
    Matrix inv() const {
        assert( m_row == m_col && "Matrix must be square" );
        size_t n = m_row;
        Matrix augmented( n, 2 * n );

        // Initialize augmented matrix: [A|I]
        for ( size_t i = 0; i < n; i++ ) {
            for ( size_t j = 0; j < n; j++ ) {
                augmented( i, j ) = ( *this )( i, j );
                augmented( i, j + n ) = ( i == j ) ? 1 : 0; // Identity matrix
            }
        }

        // Gaussian elimination
        for ( size_t i = 0; i < n; i++ ) {
            // Find the row with the maximum value in current column
            size_t maxRow = i;
            for ( size_t k = i + 1; k < n; k++ ) {
                if ( std::abs( augmented( k, i ) ) > std::abs( augmented( maxRow, i ) ) ) {
                    maxRow = k;
                }
            }

            // Swap maximum row with current row
            for ( size_t k = i; k < 2 * n; k++ ) {
                std::swap( augmented( maxRow, k ), augmented( i, k ) );
            }

            // Make all rows below this one have 0 in the current column
            for ( size_t k = i + 1; k < n; k++ ) {
                float factor = augmented( k, i ) / augmented( i, i );
                for ( size_t j = i; j < 2 * n; j++ ) {
                    augmented( k, j ) -= factor * augmented( i, j );
                }
            }
        }

        // Make upper triangular to identity
        for ( int i = n - 1; i >= 0; i-- ) {
            for ( int j = i - 1; j >= 0; j-- ) {
                float factor = augmented( j, i ) / augmented( i, i );
                for ( int k = 0; k < 2 * n; k++ ) {
                    augmented( j, k ) -= factor * augmented( i, k );
                }
            }
        }

        // Normalize diagonal
        for ( size_t i = 0; i < n; i++ ) {
            float factor = augmented( i, i );
            if ( factor == 0 ) {
                // This means the matrix isn't invertible
                throw std::runtime_error( "Matrix is not invertible" );
            }
            for ( size_t j = 0; j < 2 * n; j++ ) {
                augmented( i, j ) /= factor;
            }
        }

        // Extract the inverse from augmented matrix
        Matrix result( n, n );
        for ( size_t i = 0; i < n; i++ ) {
            for ( size_t j = 0; j < n; j++ ) {
                result( i, j ) = augmented( i, j + n );
            }
        }

        return result;
    }

    Matrix operator*( const Matrix &rhs ) const {
        assert( m_col == rhs.m_row && "Matrix size mismatch" );
        Matrix result( m_row, rhs.m_col );
        for ( int i = 0; i < m_row; ++i ) {
            for ( int j = 0; j < rhs.m_col; ++j ) {
                for ( int k = 0; k < m_col; ++k ) {
                    result( i, j ) += ( *this )( i, k ) * rhs( k, j );
                }
            }
        }
        return result;
    }

    std::vector<double> operator*( const std::vector<double> &rhs ) const {
        assert( m_col == rhs.size() && "Matrix size mismatch" );
        std::vector<double> result( m_row );
        for ( int i = 0; i < m_row; ++i ) {
            for ( int j = 0; j < m_col; ++j ) {
                result[ i ] += ( *this )( i, j ) * rhs[ j ];
            }
        }
        return result;
    }

    Matrix &operator=( const Matrix &rhs ) {
        m_row = rhs.m_row;
        m_col = rhs.m_col;
        data = rhs.data;
        return *this;
    }

    Matrix tranpose() {
        Matrix result( m_col, m_row );
        for ( int i = 0; i < m_row; ++i ) {
            for ( int j = 0; j < m_col; ++j ) {
                result( j, i ) = ( *this )( i, j );
            }
        }
        return result;
    }
    // Matrix& operator= (Matrix&& rhs) {
    //     m_row = rhs.m_row;
    //     m_col = rhs.m_col;
    //     data = std::move(rhs.data);
    //     return *this;
    // }
private:
    int m_row, m_col;
    std::vector<double> data;
};

// calculate eigenvalues via opencv
std::pair<std::vector<double>, Matrix> eig( const Matrix &m ) {
    cv::Mat cv_m( m.row(), m.col(), CV_64F );

    for ( int i = 0; i < m.row(); ++i ) {
        for ( int j = 0; j < m.col(); ++j ) {
            cv_m.at<double>( i, j ) = m( i, j );
        }
    }

    cv::Mat eigenvalue_cv, eigenvectors_cv;

    auto result = cv::eigen( cv_m, eigenvalue_cv, eigenvectors_cv );
    std::vector<double> eigenvalue( m.row() );
    Matrix eigenvectors( m.row(), m.col() );
    for ( int i = 0; i < m.row(); ++i ) {
        eigenvalue[ i ] = eigenvalue_cv.at<double>( i );
        for ( int j = 0; j < m.col(); ++j ) {
            eigenvectors( i, j ) = eigenvectors_cv.at<double>( i, j );
        }
    }
    return { eigenvalue, eigenvectors };
}

#endif // MATRIX_HPP