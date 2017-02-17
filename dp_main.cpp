#include "DeepFeatureExtractor.hpp"
#include <iostream>
#include <iomanip>

static void formatFeaturesForPCA(const vector<cv::Mat> &data, cv::Mat& dst) {
    //cv::Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
    dst.create(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32FC1);
    for(int i = 0; i < data.size(); i++) {
        //Mat Mat::reshape(int cn, int rows=0) const
        //cv::Mat image_row = data[i].clone().reshape(1,1);
        //cv::Mat row_i = dst.row(i);
        cv::Mat row_i = dst.row(i);
        //image_row.convertTo(row_i, CV_32F);
        data[i].reshape(1,1).convertTo(row_i, CV_32F);
        std:: cout << "features i" << i << " "<<  data[i].reshape(1,1) << std:: endl;
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " mean.binaryproto img.jpg" << std::endl;
        exit(1);
    }
//::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string mean_file    = argv[3];
    DeepFeatureExtractor df_extractor(model_file, trained_file, mean_file);

    string file = argv[4];

    std::cout << "----------Extraction for "
              << file << " ----------" << std::endl;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;

    //std::vector<float> d_features =  df_extractor.compute(img);
    cv::Mat d_features =  df_extractor.compute(img);
    std::cout << "Mat features:" << d_features.rows << " "<< d_features.cols << std::endl;

    std::vector<cv::Mat> vec_imgs(65, img);  
    std::vector<cv::Mat> vec_imgs_features;
    for(int i = 0; i != vec_imgs.size(); i++)
        vec_imgs_features.push_back( df_extractor.compute(vec_imgs[i]) );

    cv::Mat stack_features;
    
    formatFeaturesForPCA(vec_imgs_features, stack_features);
    
    std::cout << stack_features.rows <<" " << stack_features.cols << std::endl;

    cv::Mat out_pca_featurs;
    df_extractor.compressPCA(stack_features, out_pca_featurs, 64);

    std::vector<float> vec_features(d_features.rows * d_features.cols);
    if (d_features.isContinuous())
    {
        std::cout <<"is continuous"<<std::endl;
        vec_features.assign((float*)d_features.datastart, (float*)d_features.dataend);
    }

    
    //std::cout << out_pca_featurs <<std::endl;
    /*
    int fcols= d_features.cols, frows = d_features.rows;
    if(d_features.isContinuous())
    {
        fcols*= frows;
        frows = 1;
    }

    for(int i = 0; i < frows; i++){
        const float* fi = d_features.ptr<float>(i);
        for(int j = 0; j < fcols; j ++)
            std::cout << fi[j]; 
    }
    */
    
}
