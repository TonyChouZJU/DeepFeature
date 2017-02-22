#include "DeepFeatureExtractor.hpp"
#include <iostream>
#include <iomanip>
#include <stdlib.h>

static void formatFeaturesForPCA(const vector<cv::Mat> &data, cv::Mat& dst) {
    dst.create(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32FC1);
    for(int i = 0; i < data.size(); i++)
        data[i].copyTo(dst.row(i));
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " mean.binaryproto img.jpg" << std::endl;
        exit(1);
    }
    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string mean_file    = argv[3];
    DeepFeatureExtractor df_extractor(model_file, trained_file, mean_file);

    string file = argv[4];

    std::cout << "----------Extraction for "
              << file << " ----------" << std::endl;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;

    cv::Mat d_features =  df_extractor.compute(img);
    std::cout << "Mat features:" << d_features.rows << " "<< d_features.cols << std::endl;
    std::cout << "Image features:" << d_features << std::endl;

    /*--------------------Train --------------------*/
    /*
    std::vector<cv::Mat> vec_imgs(65, img);  
    std::vector<cv::Mat> vec_imgs_features;
    for(int i = 0; i != vec_imgs.size(); i++)
        vec_imgs_features.push_back( df_extractor.compute(vec_imgs[i]) );

    cv::Mat stack_features;
    formatFeaturesForPCA(vec_imgs_features, stack_features);
    
    cv::Mat train_pca_featurs;
    df_extractor.compressPCA(stack_features, train_pca_featurs, 64);
    std::cout << "train_pca_features size:" <<train_pca_featurs.rows << " " << train_pca_featurs.cols  <<std::endl;

    cv::Mat query_pca_features = df_extractor.projectPCA(d_features);
    std::cout << "query_pca_features size:" <<query_pca_features.rows << " " << query_pca_features.cols <<std::endl;

    */

    int n_samples = 1;
    int m_features = 2048;
    //动态开辟空间
    float* p =  new float[n_samples*m_features];        //开辟

    string dirname = "/home/zyb/VirtualDisk500/exhdd/tomorning_dataset/wonderland/cv/deep_retriver/pyTrainVideo/test_single";
    df_extractor.pictures2Features(dirname, p, n_samples);
    for(int i=0; i < m_features; i++)
            std::cout << p[i]<<" ";
    delete[] p;
}
