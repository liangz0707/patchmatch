//
// Created by LZMAC on 2016/12/22.
//

#ifndef PATCH_MATCH_PATCHMATCH_H
#define PATCH_MATCH_PATCHMATCH_H

#include <opencv2/opencv.hpp>
#include <time.h>
#include <random>
#include <vector>
#include <limits>

/*
 * 首先需要一个记录patch位置的矩阵
 * 其次需要一个通过位置提取patch的方法
 *
 * 通过上下左右更新 =》 通过多次随机更新
 *
 */

namespace lv{
    class PatchMatch{
    public:
        const static int MAX_ITR = 5;
        const static int PATCH_SIZE = 7;
        const static int STRIDE = 1;

    public:
        PatchMatch();
        PatchMatch(cv::Mat, cv::Mat, double scale = 1.0);
        PatchMatch(cv::Mat, cv::Mat,cv::Mat, cv::Mat, double scale = 1.0);

        virtual void init();//随机初始化位置；
        void init(cv::Mat ic);//使用外部矩阵初始化;

        virtual void patchmatch();
        void neighborPropgation();//临近传播
        void randomPropgation();//随即传播
        double patchDistance(const cv::Mat &, const cv::Mat &);
        void generatePatched();
        cv::Mat getCoord(); //获取成成的坐标
        cv::Mat getPatched();
        cv::Mat getCoordDist(); //获取响铃坐标的差异性；
        cv::Mat getDiff();//显示距离的热力图；

    protected:
        cv::Mat src_im; //原始图像
        cv::Mat coord;
        cv::Mat dst_im; //目标图像
        cv::Mat patched_im;
        cv::Mat diff;

        cv::Mat dst;
        cv::Mat src;

        int rows;
        int cols;
        double scale;

        bool unchanged;
    };
};


#endif //PATCH_MATCH_PATCHMATCH_H
