#include "DeepFeatureExtractor.hpp"
#include "MBS.hpp"

cv::Mat postprocessMbs(const cv::Mat &src, const cv::Mat &bwImage)
{
    // Get the contours of the connected components
    vector<vector<cv::Point>> contours;
    //findContours的输入是二值图像
    cv::findContours(bwImage,
                 contours, // a vector of contours
                 CV_RETR_EXTERNAL, // retrieve the external contours
                 CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

    //Eliminate too short or too long contours
    int cmin = 100; //minimum contour length
    std::vector<std::vector<cv::Point> >::iterator itc= contours.begin();
    while(itc!=contours.end()){
        if(itc->size()< cmin )
            itc=contours.erase(itc);
        else
            ++itc;

    }

    // Print contours' length轮廓的个数
    //std::cout << "Contours: " << contours.size() << std::endl;
    int largest_area=0;
    int largest_contour_index=0;
    cv::Rect bounding_rect;
    vector<cv::Vec4i> hierarchy;
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        double a=contourArea( contours[i],false);  //  Find the area of contour
        if(a>largest_area){
            largest_area=a;
            largest_contour_index=i;                //Store the index of largest contour
            bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
        }

    }
    float bd_rect_height = bounding_rect.height*1.0;
    float bd_rect_width = bounding_rect.width*1.0;
    //if(bd_rect_width < 100.0 || bd_rect_height <100.0 || bd_rect_width > bd_rect_height*3 || bd_rect_height > bd_rect_width*3) {
    if(bd_rect_width*bd_rect_height <= static_cast<float>(src.cols * src.rows)/64) {
        //int square_size = ((src.cols > src.rows) ? src.rows : src.cols) ;
        //bounding_rect.x = (src.cols - square_size)/2 ;
        //bounding_rect.y = (src.rows - square_size)/2 ;
        //bounding_rect.width = square_size;
        //bounding_rect.height = square_size;
       return src;
   }
   else
    // testing the bounding box
        return src(bounding_rect);
}

static void formatFeaturesForPCA(const vector<cv::Mat> &data, cv::Mat& dst) {
    dst.create(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32FC1);
    for(int i = 0; i < data.size(); i++)
        data[i].copyTo(dst.row(i));
}

void DeepFeatureExtractor::SetFeatureDims(RETRIEVER_TYPE retriver_type) {
    if(retriver_type== PAINTING_RETRIEVER){
        this->feature_dims_ = 1536;
        this->pca_dims_ = 1536;
    }
    else {
        this->feature_dims_ = 2048;
        this->pca_dims_ = 2048;
    }
}

const float* DeepFeatureExtractor::extractFeatures(const string &img_path,RETRIEVER_TYPE retriver_type) {
    SetFeatureDims(retriver_type);

    if(this->pca_dims_ > this->feature_dims_)
        return NULL;
    cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
    /*
    cv::Rect bounding_rect;
    bounding_rect.x = 40; 
    bounding_rect.y = 120; 
    bounding_rect.width = 400;
    bounding_rect.height = 400;
    img = img(bounding_rect);
    */
    cv::Mat bwImg;
    if(retriver_type !=BASE_RETRIEVER)
    {
        cv::Mat res = computeMBS(img);
        res.convertTo(bwImg,CV_8UC1);
        img = postprocessMbs(img, bwImg);
        //temp
        string::size_type found = img_path.find_last_of("/\\");
        string img_file_name = img_path.substr(found+1);
        string to_write_name = string("/home/zyb/VirtualDisk500/exhdd/tomorning_dataset/wonderland/cv/Deep_retriver_worker/tmp/online_save/")+img_file_name;
        cv::imwrite(to_write_name.c_str(), img);
    }

    if(img.empty())
        return NULL;
    const float* img_feature_ptr;
    if(retriver_type == PAINTING_RETRIEVER) {
        img_feature_ptr = this->computePainting(img);
        return img_feature_ptr;
    } 
    img_feature_ptr =  this->compute(img);
    if(this->pca_dims_ < this->feature_dims_) {
        //dont need to copy img_feature to pca_feature
        cv::Mat img_feature(1, this->feature_dims_, CV_32FC1, const_cast<float*>(img_feature_ptr));
        this->pca_feature = this->projectPCA(img_feature);
        return (float*)this->pca_feature.data;
    }
    else
        return img_feature_ptr;

    //query feature normlized is conducted in the following matching process
    //memcpy(feature, (float*)pca_feature.data, sizeof(float)*this->pca_dims_); 
}

int DeepFeatureExtractor::extractFeatures(const string &img_path, float* feature, RETRIEVER_TYPE retriver_type) {
    
    SetFeatureDims(retriver_type);

    if(this->pca_dims_ > this->feature_dims_)
        return 1;
    cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
    cv::Mat bwImg;
    if(retriver_type !=BASE_RETRIEVER)
    {
        cv::Mat res = computeMBS(img);
        res.convertTo(bwImg,CV_8UC1);
        img = postprocessMbs(img, bwImg);
    }
    if(img.empty())
        return 1;
    const float* img_feature_ptr;
    if(retriver_type == PAINTING_RETRIEVER)
        img_feature_ptr = this->computePainting(img);
    else
        img_feature_ptr = this->compute(img);
    if(this->pca_dims_ < this->feature_dims_) {
        //dont need to copy img_feature to pca_feature
        cv::Mat img_feature(1, this->feature_dims_, CV_32FC1, const_cast<float*>(img_feature_ptr));
        pca_feature = this->projectPCA(img_feature);
        memcpy(feature, (float*)this->pca_feature.data, sizeof(float)*this->pca_dims_); 
    }
    else
        memcpy(feature, img_feature_ptr, sizeof(float)*this->pca_dims_); 
    return 0;
}

int DeepFeatureExtractor::pictures2Features(const vector<string> &imgs, float* features, RETRIEVER_TYPE retriver_type) {

    SetFeatureDims(retriver_type);

    std::vector<cv::Mat> vec_imgs_features;
    vec_imgs_features.reserve(imgs.size());
    for(vector<string>::const_iterator citer = imgs.begin(); citer!=imgs.end(); citer++) {
        cv::Mat img = cv::imread(*citer, CV_LOAD_IMAGE_COLOR);
        if(img.empty()){
            if(vec_imgs_features.size()<1)
                //The First Image is Empty
                return 1;
            else
                vec_imgs_features.push_back(vec_imgs_features.back());
           continue; 
         }
            //return 1;
        const float* d_features_ptr;
        if(retriver_type == PAINTING_RETRIEVER)
            d_features_ptr = this->computePainting(img);
        else
            d_features_ptr =  this->compute(img);
        cv::Mat d_features(1, this->feature_dims_, CV_32FC1, const_cast<float*>(d_features_ptr));
        vec_imgs_features.push_back(d_features.clone());
    } 
    if(this->pca_dims_ > this-> feature_dims_ )
        return 1;
    if(this->pca_dims_ < this-> feature_dims_ && imgs.size()<this->pca_dims_ )
        return 1;

    cv::Mat stack_features;
    formatFeaturesForPCA(vec_imgs_features, stack_features);
    cv::Mat train_pca_features;
    if( this->pca_dims_ < this->feature_dims_)
        this->compressPCA(stack_features, train_pca_features, this->pca_dims_);
   else 
       train_pca_features = stack_features;
   for(int i=0; i <train_pca_features.rows ; i++) {
        //L2 normalize
        cv::normalize(train_pca_features.row(i), train_pca_features.row(i), 1, 0, cv::NORM_L2, -1); 
        memcpy(features+i*this->pca_dims_, (float*)train_pca_features.ptr<float>(i), sizeof(float)*this->pca_dims_);
   }
        //memcpy(features+offset*this->feature_dims_, (float*)d_features.data, sizeof(float)*this->feature_dims_);
   return 0;
}

int DeepFeatureExtractor::pictures2Features(string &dirname, float* features) {
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dirname.c_str())) == NULL) 
        std::cout << "Can't open" <<  dirname <<std::endl;

    int count_img = 0;
    while( (dirp = readdir(dp)) != NULL )
        //only output filename
        if(dirp->d_type == 8) {
            string file_name(dirp->d_name);
            string::size_type idx = file_name.find('.');
            string postfix_name = file_name.substr(idx+1);
            if(postfix_name!="jpg")
                break;
            
            string file_name_path = dirname + "/" + file_name; 
            cv::Mat img = cv::imread(file_name_path, CV_LOAD_IMAGE_COLOR);
            CHECK(!img.empty()) << "Unable to decode image " << file_name_path;

            const float* d_features =  this->compute(img);
            memcpy(features+count_img * this->feature_dims_, d_features, sizeof(float)*this->feature_dims_);
            count_img ++;
        }
    closedir(dp);
}

cv::Mat DeepFeatureExtractor::projectPCA(cv::InputArray vec) {
    return this->pca.project(vec);
}

void DeepFeatureExtractor::projectPCA(cv::InputArray _data, cv::OutputArray _data_reduced) {
    this->pca.project(_data, _data_reduced);
}

void DeepFeatureExtractor::compressPCA(cv::InputArray _pcaset, cv::OutputArray _compressed, int maxComponents) {
    this->pca = cv::PCA(_pcaset, //pass the data
            cv::Mat(), //there is no pre-computes mean vector
            CV_PCA_DATA_AS_ROW, //the vectors are stored as matrix rows
            maxComponents // specify how many principal components to retain
            );
    cv::Mat pcaset = _pcaset.getMat();
    _compressed.create(pcaset.rows,  maxComponents, pcaset.type());

    this->projectPCA(_pcaset, _compressed);
}

DeepFeatureExtractor::DeepFeatureExtractor(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       bool gpu_mode,
                       int gpu_id,
                       const string blob_name){
                       //int pca_dims,
if (gpu_mode)
{
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_id);
}
else
    Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    this->net_.reset(new Net<float>(model_file, TEST));
    this->net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(this->net_->num_inputs(), 1) << "Network should have exactly one input.";
    //CHECK_EQ(this->net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = this->net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    this->input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    CHECK(this->net_->has_blob(blob_name)) << " Unknown feature blob name " << blob_name
        << " in the network " << model_file;
    blob_name_ = blob_name;

    //this->feature_dims_ = net_->blob_by_name(blob_name_)->count();

    //initilize in the train or test function
    //this->feature_dims_ = pca_dims;
    //this->pca_dims_ = pca_dims;

    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    //std::cout <<"Feature dims:"<<this->feature_dims_ <<std::endl;

    /* Load the binaryproto mean file. */
    SetMean(mean_file);
}

//Specific for Monet Painting Retriever
const float* DeepFeatureExtractor::computePainting(const cv::Mat& img) {
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(input_channels);

    Preprocess(img, input_channels);

    net_->ForwardPrefilled();

    //shape (1,512, 1, 1)
    const boost::shared_ptr<Blob<float>> output_blob_1 = net_->blob_by_name("res3a_end");
    const boost::shared_ptr<Blob<float>> output_blob_2 = net_->blob_by_name("res4c_end");
    const boost::shared_ptr<Blob<float>> output_blob_3 = net_->blob_by_name("res5c_end");

    auto shape_1 = output_blob_1->shape();
    auto shape_2 = output_blob_2->shape();
    auto shape_3 = output_blob_3->shape();


    float* feature_1 = output_blob_1->mutable_cpu_data();
    float* feature_2 = output_blob_2->mutable_cpu_data();
    float* feature_3 = output_blob_3->mutable_cpu_data();

    cv::Mat featureMat_1 = cv::Mat(shape_1[1], shape_1[2]*shape_1[3], CV_32FC1, feature_1).clone();
    cv::Mat featureMat_2 = cv::Mat(shape_1[1], shape_2[1]/shape_1[1], CV_32FC1, feature_2).clone();
    cv::Mat featureMat_3 = cv::Mat(shape_1[1], shape_3[1]/shape_1[1], CV_32FC1, feature_3).clone();
    /*
    std::cout << "feature 512:"<<std::endl;
    for(int i =0; i <20; i ++)
        std::cout << featureMat_1.at<float>(i,0) <<std::endl;
    std::cout << "feature 1024:"<<std::endl;
    for(int i =0; i <10; i ++){
        std::cout << featureMat_2.at<float>(i,0) <<std::endl;
        std::cout << featureMat_2.at<float>(i,1) <<std::endl;
    }
    std::cout << "feature 2048:"<<std::endl;
    for(int i =0; i <5; i ++){
        std::cout << featureMat_3.at<float>(i,0) <<std::endl;
        std::cout << featureMat_3.at<float>(i,1) <<std::endl;
        std::cout << featureMat_3.at<float>(i,2) <<std::endl;
        std::cout << featureMat_3.at<float>(i,3) <<std::endl;
    }
    */

    //sum pooling
    cv::Mat featureMat_2_reduced;
    cv::Mat featureMat_3_reduced;

    //reduced to single column using avg
    cv::reduce(featureMat_2, featureMat_2_reduced, 1, CV_REDUCE_AVG);
    cv::reduce(featureMat_3, featureMat_3_reduced, 1, CV_REDUCE_AVG);

    /*
    std::cout << "reduced feature 1024:"<<std::endl;
    for(int i =0; i <10; i ++){
        std::cout << featureMat_2_reduced.at<float>(i,0) <<" ";
    }
    std::cout <<std::endl;

    std::cout << "reduced feature 2048:"<<std::endl;
    for(int i =0; i <10; i ++){
        std::cout << featureMat_3_reduced.at<float>(i,0) <<" ";
    }
    std::cout <<std::endl;
    */

    cv::Mat merge_feature;
    merge_feature.create(3, featureMat_1.rows*featureMat_1.cols, CV_32FC1);
    featureMat_1.reshape(1,1).copyTo(merge_feature.row(0));
    featureMat_2_reduced.reshape(1,1).copyTo(merge_feature.row(1));
    featureMat_3_reduced.reshape(1,1).copyTo(merge_feature.row(2));

    std::cout <<merge_feature <<std::endl;
    return (float*)merge_feature.data;
    
}

//cv::Mat DeepFeatureExtractor::compute(const cv::Mat& img) {
const float* DeepFeatureExtractor::compute(const cv::Mat& img) {
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(input_channels);

    Preprocess(img, input_channels);

    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    //Blob<float>* output_blob = net_->blob_by_name(blob_name_);
    const boost::shared_ptr<Blob<float>> output_blob = net_->blob_by_name(blob_name_);
    //const float* begin = output_blob->cpu_data();
    this->query_feature_ptr = output_blob->cpu_data();
    //float* begin = output_blob->mutable_cpu_data();
    //const float* end = begin + output_blob->channels();
    //cv::Mat featureMat(1, 2048, CV_32FC1, begin);
    //cv::Mat featureMat(1, this->feature_dims_ , CV_32FC1, const_cast<float*>(begin));
    //return featureMat.clone();
    return this->query_feature_ptr;
}

std::vector<std::vector<float>> DeepFeatureExtractor::compute(const std::vector<cv::Mat> &imgs) {
    std::vector<std::vector<float>> result;

    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(imgs.size(), num_channels_,
                         input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    std::vector<std::vector<cv::Mat>> input_channels_vector;
    WrapInputLayer(input_channels_vector);

    Preprocess(imgs, input_channels_vector);

    net_->ForwardPrefilled();

    //Blob<float>* output_blob = net_->blob_by_name(blob_name);
    const boost::shared_ptr<Blob<float>> output_blob = net_->blob_by_name(blob_name_);

    for(size_t i=0;i<imgs.size();++i) {
        const float *begin = output_blob->cpu_data()+i*output_blob->channels();
        const float *end = begin + output_blob->channels();
        result.push_back(std::vector<float>(begin, end));
    }
    return result;
}

/* Load the mean file in binaryproto format. */
void DeepFeatureExtractor::SetMean(const string& mean_file) {
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}
/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void DeepFeatureExtractor::WrapInputLayer(std::vector<cv::Mat>& input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    //for(int k=0;k<input_layer->num();++k) {
        for (int i = 0; i < input_layer->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }
    //}
}

void DeepFeatureExtractor::WrapInputLayer(std::vector<std::vector<cv::Mat>> &input_channels_vector) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for(int k=0;k<input_layer->num();++k) {
        std::vector<cv::Mat> input_channels;
        for (int i = 0; i < input_layer->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }
        input_channels_vector.push_back(input_channels);
    }
}

void DeepFeatureExtractor::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>& input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, input_channels);

    CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

void DeepFeatureExtractor::Preprocess(const std::vector<cv::Mat> &imgs, std::vector<std::vector<cv::Mat>> &input_channels_vector) {
    for(size_t k=0;k<imgs.size();++k){
        cv::Mat img=imgs.at(k);
        cv::Mat sample;
        if (img.channels() == 3 && num_channels_ == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4 && num_channels_ == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
        else if (img.channels() == 4 && num_channels_ == 3)
            cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
        else if (img.channels() == 1 && num_channels_ == 3)
            cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
        else
            sample = img;

        cv::Mat sample_resized;
        if (sample.size() != input_geometry_)
            cv::resize(sample, sample_resized, input_geometry_);
        else
            sample_resized = sample;

        cv::Mat sample_float;
        if (num_channels_ == 3)
            sample_resized.convertTo(sample_float, CV_32FC3);
        else
            sample_resized.convertTo(sample_float, CV_32FC1);

        cv::Mat sample_normalized;
        cv::subtract(sample_float, mean_, sample_normalized);

        cv::split(sample_normalized, input_channels_vector.at(k));
    }
}
