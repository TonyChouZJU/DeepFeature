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

using namespace caffe;
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
public:
    Classifier(const string& model_file,
               const string& trained_file,
               const string& mean_file,
               const string& label_file);

    std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

    std::vector<std::vector<Prediction> > Classify(const std::vector<cv::Mat>& imgs,int N=5);

private:
    Classifier(const Classifier&) = delete;

    Classifier& operator=(const Classifier&) = delete;

    void SetMean(const string& mean_file);

    std::vector<float> Predict(const cv::Mat& img);

    std::vector<std::vector<float> > Predict(const std::vector<cv::Mat>& imgs);

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
    cv::Mat mean_;
    std::vector<string> labels_;
};
#endif //CAFFEWORKER_CLASSIFICATION_HPP
