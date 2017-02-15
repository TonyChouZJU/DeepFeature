#include "Classifier.hpp"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " mean.binaryproto labels.txt img.jpg" << std::endl;
        exit(1);
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file   = argv[1];
    string trained_file = argv[2];
    string mean_file    = argv[3];
    string label_file   = argv[4];
    Classifier classifier(model_file, trained_file, mean_file, label_file);

    string file = argv[5];

    std::cout << "---------- Prediction for "
              << file << " ----------" << std::endl;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    cv::Mat img2 = img;

    std::vector<cv::Mat> img_vector;
    img_vector.push_back(img);
    img_vector.push_back(img2);

    if(1){
        std::vector<std::vector<Prediction> >  predictions_vector=classifier.Classify(img_vector);
        for(size_t k=0;k<predictions_vector.size();++k){
            std::vector<Prediction> predictions=predictions_vector.at(k);
            for (size_t i = 0; i < predictions.size(); ++i) {
                Prediction p = predictions[i];
                std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                          << p.first << "\"" << std::endl;
            }
            std::cout<<std::endl;
        }
    }else {
        std::vector<Prediction> predictions = classifier.Classify(img2);
        /* Print the top N predictions. */
        for (size_t i = 0; i < predictions.size(); ++i) {
            Prediction p = predictions[i];
            std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                      << p.first << "\"" << std::endl;
        }
    }
}
