#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;
int main() {
    float a[6] = {1.1,2.2, 3.4, 5.5, 0.1, 0.2};
    Mat m = Mat(2,3, CV_32F, a).clone();
    cout << m.channels()<<endl;
    cout << m.rows<<endl;
    cout <<m <<endl;

    Mat m_reduced;
    reduce(m, m_reduced, 0, CV_REDUCE_SUM);
    cout <<m_reduced.channels()<<endl;
    cout << m_reduced.rows<<endl;
    cout <<m_reduced<<endl;

    

}
