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
        const static int MAX_ITR = 10;
        const static int PATCH_SIZE = 3;
        const static int STRIDE = 1;

    public:
        PatchMatch();
        PatchMatch(cv::Mat, cv::Mat);

        void init();//随机初始化位置；

        void patchmatch();
        void neighborPropgation();//临近传播
        void randomPropgation();//随即传播
        double patchDistance(const cv::Mat &, const cv::Mat &);
        void generatePatched();
        cv::Mat getCoord(); //获取成成的坐标
        cv::Mat getPatched();

    private:
        cv::Mat src_im; //原始图像
        cv::Mat coord;
        cv::Mat dst_im; //目标图像
        cv::Mat patched_im;
        cv::Mat diff;

        int rows;
        int cols;

        bool unchanged;
    };
};


#endif //PATCH_MATCH_PATCHMATCH_H
