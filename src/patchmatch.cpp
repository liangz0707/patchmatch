//
// Created by LZMAC on 2016/12/22.
//
#include "patchmatch.h"

using namespace lv;
PatchMatch::PatchMatch(){

}

PatchMatch::PatchMatch(cv::Mat src, cv::Mat dst):src_im(src),dst_im(dst){


}

void PatchMatch::init() {
    cols = dst_im.cols;
    rows = dst_im.rows;
    diff = cv::Mat(rows,cols,CV_64FC1);
    coord = cv::Mat(rows,cols,CV_64FC2);
    patched_im = cv::Mat(rows,cols,CV_8UC3);
    srand( (unsigned)time( NULL ) );

    for(int i = 0; i < rows - PATCH_SIZE; i+=STRIDE) {
        for (int j = 0; j < cols - PATCH_SIZE; j+=STRIDE) {
            coord.at<cv::Vec2d>(i, j) = cv::Vec2d(rand() % (src_im.rows - PATCH_SIZE), rand() % (src_im.cols - PATCH_SIZE));
            cv::Vec2d pos = coord.at<cv::Vec2d>(i, j);
            cv::Mat p = dst_im.rowRange(i,i+PATCH_SIZE).colRange(j,j+PATCH_SIZE);
            cv::Mat q = src_im.rowRange(pos[0],pos[0]+PATCH_SIZE).colRange(pos[1],pos[1]+PATCH_SIZE);

            diff.at<double>(i, j) = patchDistance(p,q);
        }
    }
    unchanged = false;
}

void PatchMatch::patchmatch() {
    int itr = 0;
    while( itr++ < MAX_ITR && !unchanged) {
        unchanged = true;
        neighborPropgation();
        randomPropgation();
        if (itr % 10 == 0) {
            generatePatched();

            cv::imshow("a",patched_im);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

}
void PatchMatch::neighborPropgation() {
    for(int i = 0; i < rows - PATCH_SIZE; i += STRIDE){
        for (int j = 0; j < cols - PATCH_SIZE; j += STRIDE) {
            std::vector<cv::Vec2d> new_poses = std::vector<cv::Vec2d>{
                    coord.at<cv::Vec2d>(i + STRIDE, j) - cv::Vec2d(STRIDE,0),
                    coord.at<cv::Vec2d>(i - STRIDE, j) + cv::Vec2d(STRIDE,0),
                    coord.at<cv::Vec2d>(i, j + STRIDE) - cv::Vec2d(0,STRIDE),
                    coord.at<cv::Vec2d>(i, j - STRIDE) + cv::Vec2d(0,STRIDE),
            };
            cv::Mat p = dst_im.rowRange(i,i+PATCH_SIZE).colRange(j,j+PATCH_SIZE);

            for(auto pos: new_poses) {
                //这里需要判断一下位置是不是合理
                if(pos[0]<0 || pos[0] > src_im.rows - PATCH_SIZE || pos[1]<0 || pos[1] > src_im.cols - PATCH_SIZE) continue;

                cv::Mat q = src_im.rowRange(pos[0],pos[0]+PATCH_SIZE).colRange(pos[1],pos[1]+PATCH_SIZE);
                double new_d = patchDistance(p,q);
                if (diff.at<double>(i, j) > new_d){
                    diff.at<double>(i, j) = new_d;
                    coord.at<cv::Vec2d>(i,j) = pos;

                    unchanged = false;
                }
            }
        }
    }
}
void PatchMatch::randomPropgation() {
    for(int i = 0; i < rows - PATCH_SIZE; i += STRIDE){
        for (int j = 0; j < cols - PATCH_SIZE; j += STRIDE) {
            std::vector<cv::Vec2d> new_poses = std::vector<cv::Vec2d>{
                    coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 2,rand() % 2)- cv::Vec2d(1,1),
                    coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 4,rand() % 4)- cv::Vec2d(2,2),
                    coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 8,rand() % 8)- cv::Vec2d(4,4),
                    coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 16,rand() % 16)- cv::Vec2d(8,8),
                    coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 32,rand() % 32)- cv::Vec2d(16,16),
                    coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 64,rand() % 64)- cv::Vec2d(32,32)
            };
            cv::Mat p = dst_im.rowRange(i,i+PATCH_SIZE).colRange(j,j+PATCH_SIZE);

            for(auto pos: new_poses) {
                //这里需要判断一下位置是不是合理
                if(pos[0]<0 || pos[0] > src_im.rows - PATCH_SIZE || pos[1]<0 || pos[1] > src_im.cols - PATCH_SIZE) continue;

                cv::Mat q = src_im.rowRange(pos[0],pos[0]+PATCH_SIZE).colRange(pos[1],pos[1]+PATCH_SIZE);
                double new_d = patchDistance(p,q);
                if (diff.at<double>(i, j) > new_d){
                    diff.at<double>(i, j) = new_d;
                    coord.at<cv::Vec2d>(i,j) = pos;
                    unchanged = false;
                }
            }

        }
    }
}

void PatchMatch::generatePatched() {
    for(int i = 0; i < rows - PATCH_SIZE; i += STRIDE) {
        for(int j = 0; j < cols - PATCH_SIZE; j += STRIDE) {
            cv::Mat aim_patch = patched_im.rowRange(i,i+PATCH_SIZE).colRange(j,j+PATCH_SIZE);
            cv::Vec2d pos = coord.at<cv::Vec2d>(i,j);
            cv::Mat src_patch = src_im.rowRange(pos[0],pos[0] + PATCH_SIZE).colRange(pos[1],pos[1]+PATCH_SIZE);

            aim_patch.setTo(cv::Scalar(1,1,1));
            cv::multiply(aim_patch,src_patch,aim_patch);
        }
    }
}

double PatchMatch::patchDistance(cv::Mat q, cv::Mat p) {
    cv::Mat diff;
    cv::absdiff(q,p,diff);
    cv::multiply(diff,diff,diff);
    cv::Scalar s = cv::sum(diff);

    double dist=0;
    for(int i = 0; i< s.channels; i++){
        dist += s[i];
    }

    return dist;
}

cv::Mat PatchMatch::getPatched() {
    return patched_im;
}