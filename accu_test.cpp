#include "DeepFeatureExtractor.hpp"
#include "cospeak.hpp" 
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <chrono>
#include <fstream>

#include <algorithm>
#include <sstream>
#include <iterator>

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
    int m_features = 2048;
    DeepFeatureExtractor df_extractor(model_file, trained_file, mean_file, m_features, true);

    std::ifstream train_file("/mnt/exhdd/tomorning_dataset/wonderland/cv/Deep_retriver_worker/tmp/train_file_list.txt", ios::in);
    string s;
    vector<string> train_file_vec;
    vector<string> train_label_vec;
    while(getline(train_file, s)) {
        string::size_type position = s.find(" ");
        if(position == s.npos)
            std::cout <<"not found"<< std::endl;
        else{
            train_file_vec.push_back(s.substr(0, position));
            train_label_vec.push_back(s.substr(position+1));
            //std::cout << s.substr(0, position) << s.substr(position) <<std::endl;
        }
    }
    std::cout <<"load train file list" <<std::endl;

    int n_samples = train_file_vec.size();
    //动态开辟空间
    float* dataset_ptr =  new float[n_samples*m_features];  

    /*--------------------Train --------------------*/
    typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
    TimePoint start = std::chrono::system_clock::now();
    df_extractor.pictures2Features(train_file_vec, dataset_ptr);
    std::cout<<"Training time:" << std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now() - start).count()<<"\n";


    std::ifstream test_file("/mnt/exhdd/tomorning_dataset/wonderland/cv/Deep_retriver_worker/tmp/test_file_list.txt", ios::in);
    string s_test;
    string query_image_path;
    string test_label;
    float total = 0.0;
    float correct = 0.0; 
    float all_time = 0.0;
    while(getline(test_file, s_test)) {
        string::size_type position = s_test.find(" ");
        if(position == s_test.npos)
            std::cout <<"not found"<< std::endl;
        else{
            query_image_path = s_test.substr(0, position);
            test_label = s_test.substr(position+1);
            start = std::chrono::system_clock::now();
            const float* query_ptr = df_extractor.extractFeatures(query_image_path);
            std::pair<int, float> query_result = wildcard_test(query_ptr, dataset_ptr, n_samples, m_features); 
            string query_label = train_label_vec[query_result.first];
            all_time += std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now() - start).count();

            if(query_result.second >= 0.5 && test_label==query_label)
                correct+=1.0;
            else
                std::cout << test_label << "vs" <<query_label <<std::endl;
            total += 1.0;
        }
    }
    std:: cout <<"All accuracy:"<<correct/total <<std::endl;
    std:: cout << "average test time:" <<all_time/total<< std::endl;
    delete[] dataset_ptr;

}
