#ifndef SLIC_HPP
#define SLIC_HPP

#include "matrix.hpp"
#include <cmath>
#include <limits>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>

// function to find minimum gradient point at 3 by 3 neighborhood
std::pair<int, int> minimum_gradient( cv::Mat &img, int x, int y ) {
    cv::Mat gray_img;
    cv::cvtColor( img, gray_img, CV_BGR2GRAY );
    int new_x = x;
    int new_y = y;
    double min_grad = std::numeric_limits<double>::max();

    for ( int i = x - 1; i <= x + 1; ++i ) { // 가로방향 3칸
        for ( int j = y - 1; j <= y + 1; ++j ) {
            double current = gray_img.at<double>( j, i );
            double right = gray_img.at<double>( j + 1, i );
            double down = gray_img.at<double>( j, i + 1 );

            double grad = std::sqrt( std::pow( current - right, 2 ) + std::pow( current - down, 2 ) );

            if ( grad < min_grad ) {
                min_grad = grad;
                new_x = i;
                new_y = j;
            }
        }
    }

    return { new_x, new_y };
}

// function to calculate distance including not only color but also spacial values.

double distance( cv::Mat &img, int S, std::pair<int, int> pt1, std::pair<int, int> pt2 ) {
    // Euclidean Distance between pixel
    double d_s = std::sqrt( static_cast<double>( pt1.first - pt2.first ) * static_cast<double>( pt1.first - pt2.first ) / static_cast<double>( img.rows ) / static_cast<double>( img.rows ) + static_cast<double>( pt1.second - pt2.second ) * ( pt1.second - pt2.second ) / static_cast<double>( img.cols ) / static_cast<double>( img.cols ) );

    // Color-space distance
    auto pixel1 = img.at<cv::Vec3b>( pt1.second, pt1.first );
    auto pixel2 = img.at<cv::Vec3b>( pt2.second, pt2.first );

    double d_c = std::sqrt( static_cast<double>( pixel1[ 0 ] - pixel2[ 0 ] ) * static_cast<double>( pixel1[ 0 ] - pixel2[ 0 ] ) + static_cast<double>( pixel1[ 1 ] - pixel2[ 1 ] ) * static_cast<double>( pixel1[ 1 ] - pixel2[ 1 ] ) + static_cast<double>( pixel1[ 2 ] - pixel2[ 2 ] ) * static_cast<double>( pixel1[ 2 ] - pixel2[ 2 ] ) ) / 255.0;

    double d = std::sqrt( std::pow( d_s, 2 ) + std::pow( d_c / std::sqrt( 30 ), 2 ) );
    return d;
}

// function to define affinity between two nodes(superpixel on this example)
double affinity( cv::Mat &img, std::pair<int, int> pt1, std::pair<int, int> pt2, cv::Vec3b pixel1, cv::Vec3b pixel2 ) {
    // color distance
    double d_c = std::sqrt( static_cast<double>( pixel1[ 0 ] - pixel2[ 0 ] ) * static_cast<double>( pixel1[ 0 ] - pixel2[ 0 ] ) + static_cast<double>( pixel1[ 1 ] - pixel2[ 1 ] ) * static_cast<double>( pixel1[ 1 ] - pixel2[ 1 ] ) + static_cast<double>( pixel1[ 2 ] - pixel2[ 2 ] ) * static_cast<double>( pixel1[ 2 ] - pixel2[ 2 ] ) ) / 255.0;

    // spatial distance
    auto [ x1, y1 ] = pt1;
    auto [ x2, y2 ] = pt2;
    double d_s = std::sqrt( static_cast<double>( pt1.first - pt2.first ) * static_cast<double>( pt1.first - pt2.first ) / static_cast<double>( img.rows ) / static_cast<double>( img.rows ) + static_cast<double>( pt1.second - pt2.second ) * ( pt1.second - pt2.second ) / static_cast<double>( img.cols ) / static_cast<double>( img.cols ) );

    double d = d_c / std::sqrt( 3 ) + d_s / std::sqrt( 20 );
    d = std::exp( -d );

    return d;
}

std::map<std::pair<int, int>, int> slic( cv::Mat &image, std::map<std::string, int> options = { { "iter", 20 }, { "k", 400 } } ) {
    auto [ width, height ] = image.size();
    int k = options[ "k" ];          // number of superpixels
    int Sx = width / std::sqrt( k ); // 평균적인 superpixel별 픽셀 수 -> 초기에는 S간격 grid로 clustering
    int Sy = height / std::sqrt( k );

    // 1. First, apply SLIC algorithm to build superpixels
    // Initialize cluster centers with grid and move cluster centers to lower
    // gradient position in 3 x 3 neighborhood

    std::vector<std::pair<int, int>> cluster_coord;
    std::vector<cv::Vec3b> cluster_colors;

    for ( int i = Sx / 2; i < width - Sx / 2; i += Sx ) {
        for ( int j = Sy / 2; j < height - Sy / 2; j += Sy ) {
            auto [ new_x, new_y ] = minimum_gradient( image, i, j );
            cluster_coord.push_back( { new_x, new_y } );
            cluster_colors.push_back( image.at<cv::Vec3b>( new_y, new_x ) );
        }
    }

    k = cluster_coord.size();

    std::map<std::pair<int, int>, int> labels;
    std::map<std::pair<int, int>, double> distances;

    for ( int i = 0; i < width; ++i ) {
        for ( int j = 0; j < height; ++j ) {
            labels[ { i, j } ] = -1;
            distances[ { i, j } ] = std::numeric_limits<double>::max();
        }
    }

    // 2. Assign each pixel to the nearest cluster center
    for ( int iter = 0; iter < options[ "iter" ]; ++iter ) {
        // == Assignment Step ==
        // 1. look 2S x 2S around each cluster
        // 2. if distance between center is lower than min_distance(j,i),
        //    update it by labeling as k-th center, and renew min_distance.
        for ( int c = 0; c < k; ++c ) {
            for ( int dx = -Sx - 2; dx <= Sx + 2; ++dx ) {
                for ( int dy = -Sy - 2; dy <= Sy + 2; ++dy ) {
                    auto [ x, y ] = cluster_coord[ c ];
                    x += dx;
                    y += dy;

                    if ( x < 0 || x >= width || y < 0 || y >= height )
                        continue;

                    double d = distance( image, ( Sx + Sy ) / 2, cluster_coord[ c ], { x, y } );
                    if ( d < distances[ { x, y } ] ) {
                        distances[ { x, y } ] = d;
                        labels[ { x, y } ] = c;
                    }
                }
            }
        }

        // == Update(MoveMean) step ==
        // 1. compute new cluster centers : mean_x, mean_y
        std::vector<int> cluster_size( k, 0 );
        std::vector<std::array<int, 5>> cluster_xyrgb( k, { 0, 0, 0, 0, 0 } );

        for ( int i = 0; i < width; ++i ) {
            for ( int j = 0; j < height; ++j ) {
                int c = labels[ { i, j } ];
                if ( c < 0 )
                    continue;
                cluster_size[ c ] += 1;
                cluster_xyrgb[ c ][ 0 ] += i;
                cluster_xyrgb[ c ][ 1 ] += j;
                cluster_xyrgb[ c ][ 2 ] += image.at<cv::Vec3b>( j, i )[ 0 ];
                cluster_xyrgb[ c ][ 3 ] += image.at<cv::Vec3b>( j, i )[ 1 ];
                cluster_xyrgb[ c ][ 4 ] += image.at<cv::Vec3b>( j, i )[ 2 ];
            }
        }

        for ( int c = 0; c < k; ++c ) {
            if ( cluster_size[ c ] == 0 )
                continue;
            int mean_x = cluster_xyrgb[ c ][ 0 ] / cluster_size[ c ];
            int mean_y = cluster_xyrgb[ c ][ 1 ] / cluster_size[ c ];
            uchar mean_r = static_cast<uchar>( cluster_xyrgb[ c ][ 2 ] / cluster_size[ c ] );
            uchar mean_g = static_cast<uchar>( cluster_xyrgb[ c ][ 3 ] / cluster_size[ c ] );
            uchar mean_b = static_cast<uchar>( cluster_xyrgb[ c ][ 4 ] / cluster_size[ c ] );
            cluster_coord[ c ] = { mean_x, mean_y };
            cluster_colors[ c ] = cv::Vec3b( { mean_r, mean_g, mean_b } );
        }
    }

    // plot superpixel
    cv::Mat result = image.clone();
    cv::Mat result2 = image.clone();
    for ( int i = 0; i < width; ++i ) {
        for ( int j = 0; j < height; ++j ) {
            if ( i == 0 || j == 0 || i == width - 1 || j == height - 1 ) {
                result2.at<cv::Vec3b>( j, i ) = cluster_colors[ labels[ { i, j } ] ];
                continue;
            }
            result2.at<cv::Vec3b>( j, i ) = cluster_colors[ labels[ { i, j } ] ];
            int c = labels[ { i, j } ];
            int c_r = labels[ { i + 1, j } ];
            int c_l = labels[ { i - 1, j } ];
            int c_u = labels[ { i, j + 1 } ];
            int c_d = labels[ { i, j - 1 } ];

            if ( c != c_r || c != c_l || c != c_u || c != c_d ) {
                result.at<cv::Vec3b>( j, i ) = cv::Vec3b( { 0, 0, 255 } );
            }
        }
    }
    cv::imwrite( "result.png", result );
    cv::imwrite( "result2.png", result2 );

    Matrix affinity_matrix( k, k );
    for ( int i = 0; i < k; ++i ) {
        for ( int j = i; j < k; ++j ) {
            affinity_matrix( i, j ) = affinity( image, cluster_coord[ i ], cluster_coord[ j ], cluster_colors[ i ], cluster_colors[ j ] );
            affinity_matrix( j, i ) = affinity_matrix( i, j );
        }
    }

    Matrix D( k, k );
    Matrix D_sqrt( k, k );
    Matrix D_sqrt_inverse( k, k );
    for ( int i = 0; i < k; ++i ) {
        for ( int j = 0; j < k; ++j ) {
            D( i, i ) += affinity_matrix( i, j );
        }
        D_sqrt( i, i ) = std::pow( D( i, i ), 0.5 );
        D_sqrt_inverse( i, i ) = std::pow( D( i, i ), -0.5 );
    }

    Matrix L = D_sqrt_inverse * ( D - affinity_matrix ) * D_sqrt_inverse;
    std::vector<double> asd( k, 1.0 );
    auto asdf = L * D_sqrt * asd;

    auto [ D_, V ] = eig( L );
    std::vector<double> second_eigvector( k );
    for ( int i = 0; i < k; ++i ) {
        second_eigvector[ i ] = V( k - 2, i );
    }
    auto z = D_sqrt * second_eigvector;

    std::vector<int> cluster_labels_ncut( k, 0 );
    for ( int i = 0; i < k; ++i ) {
        if ( z[ i ] >= 0 )
            cluster_labels_ncut[ i ] = 1;
        else
            cluster_labels_ncut[ i ] = 0;
    }

    cv::Mat result3 = image.clone();
    cv::Mat filter = image.clone();
    cv::Mat result4;
    for ( int i = 0; i < width; ++i ) {
        for ( int j = 0; j < height; ++j ) {
            if ( cluster_labels_ncut[ labels[ { i, j } ] ] == 0 ) {
                result3.at<cv::Vec3b>( j, i ) = cv::Vec3b( { 255, 0, 30 } );
                filter.at<cv::Vec3b>( j, i ) = cv::Vec3b( { 6, 6, 6 } );
            } else {
                result3.at<cv::Vec3b>( j, i ) = cv::Vec3b( { 0, 255, 30 } );
            }
        }
    }
    cv::imwrite( "result3.png", result3 );

    cv::addWeighted( image, 0.4, filter, 0.6, 0.0, result4 );

    cv::imwrite( "result4.png", result4 );

    return labels;
}

#endif // SLIC_HPP