#include "DeepFeatureExtractor.hpp"
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <chrono>

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
    int n_samples = 1;
    int m_features = 2048;
    DeepFeatureExtractor df_extractor(model_file, trained_file, mean_file, m_features);

    //动态开辟空间
    float* p =  new float[n_samples*m_features];  

    /*--------------------Train --------------------*/
    string test_base_dir= argv[4];
    std::vector<string> vec_imgs;
    for(int i = 0; i < n_samples; i++)
        vec_imgs.push_back(test_base_dir + "/test_" + std::to_string(i) + ".jpg");

    df_extractor.pictures2Features(vec_imgs, p);

    typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
    for(int i=0;i< 100; i++){
    TimePoint start = std::chrono::system_clock::now();
    const float* q_ptr = df_extractor.extractFeatures(vec_imgs[0]);
    std::cout<<std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now() - start).count()<<"\n";
    }


}
