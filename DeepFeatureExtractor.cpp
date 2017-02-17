#include "DeepFeatureExtractor.hpp"


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
                       const string blob_name) {
if (gpu_mode)
    Caffe::set_mode(Caffe::GPU);
else
    Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    CHECK(net_->has_blob(blob_name)) << " Unknown feature blob name " << blob_name
        << " in the network " << model_file;
    blob_name_ = blob_name;

    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    /* Load the binaryproto mean file. */
    SetMean(mean_file);
}

//std::vector<float> DeepFeatureExtractor::compute(const cv::Mat& img) {
cv::Mat DeepFeatureExtractor::compute(const cv::Mat& img) {
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(input_channels);

    Preprocess(img, input_channels);

    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    //Blob<float>* output_blob = net_->blob_by_name(blob_name_);
    const boost::shared_ptr<Blob<float>> output_blob = net_->blob_by_name(blob_name_);
    //const float* begin = output_blob->cpu_data();
    float* begin = output_blob->mutable_cpu_data();
    const float* end = begin + output_blob->channels();
    cv::Mat featureMat(1, 2048, CV_32FC1, begin);
    return featureMat.clone();
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
