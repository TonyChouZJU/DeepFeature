#ifndef COSPEAK_HPP_
#define COSPEAK_HPP_

//#include <vector>
#include <utility>

/*class cospeak_t{
    private:
    void *train_hook;
    public:
    cospeak_t(const std::vector< std::vector<float> > &train);
    ~cospeak_t();
    std::pair<int, float> simple_test(const std::vector<float> &query) const;
};*/

std::pair<int, float> wildcard_test(const float *const query, const float *const train, const unsigned int rows, const unsigned int cols);

#endif /* COSPEAK_HPP_ */
