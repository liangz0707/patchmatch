#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/patchmatch.h"
using namespace lv;
using namespace std;
int main() {
    cv::Mat a = cv::imread("a.png");
    cv::Mat b = cv::imread("b.png");
    /*
    cv::imshow("a",a);
    cv::imshow("b",b);
    cv::destroyAllWindows();
    cout<<"a:"<<a.rows<<" "<<a.cols<<endl;
    cout<<"b:"<<b.rows<<" "<<b.cols<<endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    */
    PatchMatch pm(a,b);
    pm.init();
    pm.patchmatch();


    return 0;
}