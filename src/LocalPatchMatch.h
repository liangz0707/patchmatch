//
// Created by LZMAC on 2017/1/10.
//

#ifndef PATCH_MATCH_LOCALPATCHMATCH_H
#define PATCH_MATCH_LOCALPATCHMATCH_H
#include "patchmatch.h"
#include "math.h"

using  namespace lv;
/**
 * 目前实现的版本需要保证src和dst的大小是一样的；
 */
class LocalPatchMatch : public PatchMatch{
public:
    LocalPatchMatch();
    LocalPatchMatch(cv::Mat, cv::Mat, double scale = 1.0, int local_win = 10);
    LocalPatchMatch(cv::Mat, cv::Mat,cv::Mat, cv::Mat, double scale = 1.0, int local_win = 10);

    virtual void init();//随机初始化位置；
    virtual void patchmatch();

    cv::Mat drawCoord();

    void localPropgation();//局部搜索
protected:
    int local_win;//局部搜索窗口的大小；
};


#endif //PATCH_MATCH_LOCALPATCHMATCH_H
