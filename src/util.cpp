#include "util.hpp"

namespace Util {
// Compute the norm of a vector
// @param v vector
// @return norm of v
double norm( std::vector<double> &v ) {
    return std::sqrt( std::inner_product( v.begin(), v.end(), v.begin(), 0.0 ) );
}

// Generate a random number in [min, max]
// @param min lower bound
// @param max upper bound
// @return random number in [min, max]
double rand( double min, double max ) {
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( min, max );
    return dis( gen );
}

} // namespace Util