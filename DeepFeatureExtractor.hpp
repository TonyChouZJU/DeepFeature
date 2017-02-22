#ifndef CAFFEWORKER_CLASSIFICATION_HPP
#define CAFFEWORKER_CLASSIFICATION_HPP
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

using namespace caffe;
using std::string;


class DeepFeatureExtractor {
public:
    DeepFeatureExtractor(const string& model_file,
               const string& trained_file,
               const string& mean_file,
               bool gpu_mode=true,
               int gpu_id=0,
               const string blob_name="pool5");

    //std::vector<float> compute(const cv::Mat& img);
    cv::Mat compute(const cv::Mat& img);

    std::vector<std::vector<float> > compute(const std::vector<cv::Mat>& imgs);

    void compressPCA(cv::InputArray _pcaset, cv::OutputArray _compressed, int maxComponents);

    cv::Mat projectPCA(cv::InputArray vec);

    void projectPCA(cv::InputArray _data, cv::OutputArray _data_reduced);

    int pictures2Features(string &dirname, float* features, unsigned int len);
    int pictures2Features(const vector<string> &dirname, float* features, unsigned int len);

private:
    DeepFeatureExtractor(const DeepFeatureExtractor&) = delete;

    DeepFeatureExtractor& operator=(const DeepFeatureExtractor&) = delete;

    void SetMean(const string& mean_file);

    void WrapInputLayer(std::vector<cv::Mat>& input_channels);

    void WrapInputLayer(std::vector<std::vector<cv::Mat>>& input_channels_vector);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>& input_channels);

    void Preprocess(const std::vector<cv::Mat>& imgs,
                    std::vector<std::vector<cv::Mat>> &input_channels);


private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    int feature_dims_;
    cv::Mat mean_;
    string blob_name_;
    cv::PCA pca; 

};
#endif //CAFFEWORKER_CLASSIFICATION_HPP
