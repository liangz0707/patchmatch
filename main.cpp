#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/patchmatch.h"
#include <sstream>
#include "src/LocalPatchMatch.h"
using namespace lv;
using namespace std;
using namespace cv;

std::vector<cv::Mat> readImages() {
    std::vector<cv::Mat> images;
    stringstream ss;
    string path = "../res/bike/00";

    for (int i = 10; i < 20; i++) {
        ss.str("");
        ss<<path<<i<<".jpg";
        string s=ss.str();
        cout<<s<<endl;
        images.push_back(imread(s));
    }
    cout<<"共读取图片"<<images.size()<<endl;
    return images;
}

std::vector<cv::Mat> downsampleImages(const std::vector<cv::Mat> & images) {
    std::vector<cv::Mat> downsample_images;

    downsample_images.reserve(images.size());
    for(cv::Mat image :images) {
        cv::Mat downsample_image;
        cv::resize(image,downsample_image,Size(0,0),1./3,1./3);
        downsample_images.push_back(downsample_image);
    }
    return downsample_images;
}

int main() {
    std::vector<cv::Mat> images = readImages();
    std::vector<cv::Mat> downsample_images = downsampleImages(images);

    LocalPatchMatch pm0(downsample_images[5],downsample_images[6]);
    LocalPatchMatch pm1(downsample_images[7],downsample_images[6]);

    pm0.init();
    pm0.patchmatch();

    pm1.init();
    //pm1.patchmatch();

    cv::Mat coord_dist0 = pm0.getCoordDist();
    cv::Mat coord_dist1 = pm1.getCoordDist();

    cv::Mat coord_dist = Mat_<double>(coord_dist0+coord_dist1)(Rect(0,0,coord_dist0.cols-30,coord_dist0.rows-30));

    cv::normalize(coord_dist,coord_dist,0,1,cv::NORM_MINMAX);

    Mat cord = pm0.drawCoord();
    Mat diff = pm0.getDiff();


    imshow("coord_dist",coord_dist);
    imshow("cord",cord);
    waitKey(0);


    return 0;
}