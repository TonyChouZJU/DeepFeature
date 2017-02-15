#include "DeepFeatureExtractor.hpp"
#include <iostream>
#include <iomanip>

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

    std::vector<float> d_features =  df_extractor.compute(img);
    for(int i = 0; i !=d_features.size(); i++)
        std::cout<< d_features[i] <<std::endl;
    
}
