#include"MBS.hpp"
#include<unistd.h>
#include<ctime>
using namespace cv;  
using namespace std;

cv::Mat postprocessMbs_local(const cv::Mat &src)
{
    cv::Mat bwImage;
    //cv::cvtColor(res, bwImage, CV_RGB2GRAY);
    src.convertTo(bwImage,CV_8UC1);

    // Get the contours of the connected components
    vector<vector<Point>> contours;
    //findContours的输入是二值图像
    findContours(bwImage,
                 contours, // a vector of contours
                 CV_RETR_EXTERNAL, // retrieve the external contours
                 CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours

    // Print contours' length轮廓的个数
    cout << "Contours: " << contours.size() << endl;

    int largest_area=0;
    int largest_contour_index=0;
    Rect bounding_rect;
    vector<Vec4i> hierarchy;
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        double a=contourArea( contours[i],false);  //  Find the area of contour
        if(a>largest_area){
            largest_area=a;
            largest_contour_index=i;                //Store the index of largest contour
            bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
        }

    }

    // testing the bounding box
    Rect r0= boundingRect(Mat(contours[0]));//boundingRect获取这个外接矩形
    return src(r0);
}
int main(int argc,char *argv[])
{

    Mat im=imread("/home/zyb/cv/saliency/zyb_mbs/dog.jpg");

    Mat res = computeMBS(im);
    res = postprocessMbs_local(res);
    cv::imwrite("./dog_saliency_roi.jpg", res);

    return 0;
}
