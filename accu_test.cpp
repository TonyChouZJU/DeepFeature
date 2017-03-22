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

#include <map>


bool LoadDLRetrieverModel(const string &model_path, float * model_data,int DLRetrieverTrainFeatureCount, int DLRetrieverImageFeatureSize){
    std::ifstream fin(model_path, std::ios::in);
    for (size_t i = 0;i < DLRetrieverImageFeatureSize * DLRetrieverTrainFeatureCount;i++) {
        if (!(fin >> model_data[i])) {
            if (i != DLRetrieverImageFeatureSize * DLRetrieverTrainFeatureCount - 1) {
                std::cout<<"LoadDLRetrieverModel1 "<<model_path<<" failed: data format error" <<std::endl;
            return false;
            }
        }
    }

    return true;
}



bool SaveDLRetrieverModel(const float *model_data,const string &model_path, int DLRetrieverTrainFeatureCount, int DLRetrieverImageFeatureSize) {
    std::ofstream fout(model_path, std::ios::out);
	for (size_t i = 0;
		i < DLRetrieverImageFeatureSize * DLRetrieverTrainFeatureCount;i++) {
		fout << model_data[i] << " ";
	}
	fout.close();
	return true;
}


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " mean.binaryproto" << std::endl;
        exit(1);
    }
    //::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string mean_file    = argv[3];
    int m_features = 2048;
    DeepFeatureExtractor df_extractor(model_file, trained_file, mean_file, m_features, true, 0);

    std::ifstream train_file("/mnt/exhdd/tomorning_dataset/wonderland/cv/Deep_retriver_worker/tmp/online_test_3/train_file_list.txt", ios::in);
    string s;
    vector<string> train_file_vec;
    vector<string> train_label_vec;
    while(getline(train_file, s)) {
        string::size_type position = s.find_last_of(" ");
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

    typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;
    TimePoint start = std::chrono::system_clock::now();
    string loadingModelFile("/mnt/exhdd/tomorning_dataset/wonderland/cv/Deep_retriver_worker/tmp/online_test_3/train_features_file.txt");
    if(!LoadDLRetrieverModel(loadingModelFile, dataset_ptr, n_samples, m_features)) {
        std::cout <<"Start Training......."<<std::endl;
        //--------------------Train --------------------
        df_extractor.pictures2Features(train_file_vec, dataset_ptr);
        std::cout<<"Training time:" << std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now() - start).count()<<"\n";
        if(SaveDLRetrieverModel(dataset_ptr,loadingModelFile, n_samples, m_features)) 
            std::cout <<"Model has saved!"<<std::endl;
        else
            std::cout <<"Error!!Model not saved!"<<std::endl;

    }

    std::ifstream test_file("/mnt/exhdd/tomorning_dataset/wonderland/cv/Deep_retriver_worker/tmp/online_test_3/test_file_list.txt", ios::in);
    string s_test;
    string query_image_path;
    string test_label;
    float total = 0.0;
    float correct = 0.0; 
    float all_time = 0.0;
    std::map<string, float> map_correct; 
    std::map<string, float> map_count; 
    while(getline(test_file, s_test)) {
        string::size_type position = s_test.find_last_of(" ");
        if(position == s_test.npos)
            std::cout <<"not found"<< std::endl;
        else{
            query_image_path = s_test.substr(0, position);
            test_label = s_test.substr(position+1);
            if(map_count.find(test_label) == map_count.end())
            {
                map_count[test_label] = 0.0;
                map_correct[test_label] = 0.0;
            }
            else
                map_count[test_label] += 1.0;
            start = std::chrono::system_clock::now();
            try {
                const float* query_ptr = df_extractor.extractFeatures(query_image_path, IM_RETRIEVER);
                std::pair<int, float> query_result = wildcard_test(query_ptr, dataset_ptr, n_samples, m_features); 
                string query_label = train_label_vec[query_result.first];
                all_time += std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now() - start).count();

                if(query_result.second >= 0.6 && test_label==query_label)
                {
                    correct+=1.0;
                    map_correct[test_label] += 1.0;
                }
                //else
                //{
                    //std::cout << test_label << "vs" <<query_label <<"       ";
                    //std::cout <<"This test:"<<s_test<< std::endl;
                //}
            }
            catch(std::exception& e){
                std::cout << e.what()<< std::endl;
            }
            total += 1.0;
        }
    }

    std::map<string, float>::iterator m_it_correct = map_correct.begin();
    std::map<string, float>::iterator m_it_count = map_count.begin();
    for( ;m_it_correct != map_correct.end(); m_it_correct ++, m_it_count++)
    {
        std::cout <<"Label:"<<m_it_correct->first << ";" << m_it_correct->second / m_it_count->second<< std::endl;
    }

    std:: cout <<"All accuracy:"<<correct/total <<std::endl;
    std:: cout <<"All correct:"<<correct <<std::endl;
    std:: cout <<"Total:"<<total <<std::endl;
    std:: cout << "average test time:" <<all_time/total<< std::endl;
    delete[] dataset_ptr;

}
