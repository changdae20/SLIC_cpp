#include "util.hpp"
#include <fmt/core.h>

#include "matrix.hpp"
#include "slic.hpp"
#include <iostream>
#include <string>
int main( int argc, char *argv[] ) {
    auto img = cv::imread( argv[ 1 ] );

    auto labels = slic( img, { { "iter", 10 },
                               { "k", 441 } } );

    return 0;
}