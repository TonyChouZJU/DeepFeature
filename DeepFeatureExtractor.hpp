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
               int pca_dims,
               bool gpu_mode=true,
               int gpu_id=0,
               const string blob_name="pool5");


    void compressPCA(cv::InputArray _pcaset, cv::OutputArray _compressed, int maxComponents);

    cv::Mat projectPCA(cv::InputArray vec);

    void projectPCA(cv::InputArray _data, cv::OutputArray _data_reduced);

    int pictures2Features(string &dirname, float* features);
    int pictures2Features(const vector<string> &imgs, float* features);

    int extractFeatures(const string &img_path, float* feature);
    const float* extractFeatures(const string &img_path);

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

    //cv::Mat compute(const cv::Mat& img);
    const float* compute(const cv::Mat& img);

    std::vector<std::vector<float> > compute(const std::vector<cv::Mat>& imgs);


private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    int feature_dims_;
    int pca_dims_;
    cv::Mat mean_;
    string blob_name_;
    cv::PCA pca; 
    const float* query_feature_ptr;
    cv::Mat pca_feature;

};
static void formatFeaturesForPCA(const vector<cv::Mat> &data, cv::Mat& dst); 
#endif //CAFFEWORKER_CLASSIFICATION_HPP
