//
// Created by LZMAC on 2016/12/22.
//
#include "patchmatch.h"

using namespace lv;
PatchMatch::PatchMatch(){

}

PatchMatch::PatchMatch(cv::Mat src, cv::Mat dst, double scale):src(src),dst(dst),src_im(src),dst_im(dst),scale(scale){}
PatchMatch::PatchMatch(cv::Mat src, cv::Mat dst,cv::Mat src_im, cv::Mat dst_im, double scale):src(src),dst(dst),src_im(src_im),dst_im(dst_im),scale(scale){}

void PatchMatch::init() {
    cols = dst_im.cols;
    rows = dst_im.rows;
    diff = cv::Mat(rows,cols,CV_64FC1);
    coord = cv::Mat(rows,cols,CV_64FC2);
    patched_im = cv::Mat::zeros(rows,cols,CV_8UC3);
    srand( (unsigned)time( NULL ) );

    for(int i = 0; i < rows - PATCH_SIZE; i+=STRIDE) {
        for (int j = 0; j < cols - PATCH_SIZE; j+=STRIDE) {
            coord.at<cv::Vec2d>(i, j) = cv::Vec2d(rand() % (src_im.rows - PATCH_SIZE), rand() % (src_im.cols - PATCH_SIZE));
            //coord.at<cv::Vec2d>(i, j) = cv::Vec2d(i, j);

            cv::Vec2d pos = coord.at<cv::Vec2d>(i, j);
            cv::Mat p = dst_im.rowRange(i,i+PATCH_SIZE).colRange(j,j+PATCH_SIZE);
            cv::Mat q = src_im.rowRange(pos[0],pos[0]+PATCH_SIZE).colRange(pos[1],pos[1]+PATCH_SIZE);

            diff.at<double>(i, j) = patchDistance(p,q);
        }
    }
    unchanged = false;
}

void PatchMatch::patchmatch() {
    //cv::namedWindow("result");
    int itr = 0;
    while( itr ++ < MAX_ITR){
        unchanged = true;
        time_t a= clock();
        neighborPropgation();
        randomPropgation();
        time_t b = clock();
    }
    //cv::destroyAllWindows();
}
void PatchMatch::neighborPropgation() {
    cv::Mat p;
    cv::Mat q;
    double new_d;
    std::vector<cv::Vec2d> new_poses;
    new_poses.reserve(4);
    new_poses.resize(4);
    for(int i = 0; i < rows - PATCH_SIZE; i += STRIDE){
        for (int j = 0; j < cols - PATCH_SIZE; j += STRIDE) {

            new_poses[0]=coord.at<cv::Vec2d>(i + STRIDE, j) - cv::Vec2d(STRIDE,0);
            new_poses[1]=coord.at<cv::Vec2d>(i - STRIDE, j) + cv::Vec2d(STRIDE,0);
            new_poses[2]=coord.at<cv::Vec2d>(i, j + STRIDE) - cv::Vec2d(0,STRIDE);
            new_poses[3]=coord.at<cv::Vec2d>(i, j - STRIDE) + cv::Vec2d(0,STRIDE);

            p = dst_im.rowRange(i,i+PATCH_SIZE).colRange(j,j+PATCH_SIZE);

            for(auto pos: new_poses) {
                //这里需要判断一下位置是不是合理
                if(isnan(pos[0]) ||  pos[0]<0 || pos[0] > src_im.rows - PATCH_SIZE || pos[1]<0 || pos[1] > src_im.cols - PATCH_SIZE) continue;

                q = src_im.rowRange(pos[0],pos[0]+PATCH_SIZE).colRange(pos[1],pos[1]+PATCH_SIZE);
                new_d = patchDistance(p,q);
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
    cv::Mat p;
    cv::Mat q;
    double new_d;
    std::vector<cv::Vec2d> new_poses;
    new_poses.reserve(7);
    new_poses.resize(7);
    for(int i = 0; i < rows - PATCH_SIZE; i += STRIDE){
        for (int j = 0; j < cols - PATCH_SIZE; j += STRIDE) {

            //new_poses[0]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 4,rand() % 4)- cv::Vec2d(2,2);
            new_poses[1]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 8,rand() % 8)- cv::Vec2d(4,4);
            new_poses[2]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 16,rand() % 16)- cv::Vec2d(8,8);
            new_poses[3]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 32,rand() % 32)- cv::Vec2d(16,16);
            new_poses[4]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 64,rand() % 64)- cv::Vec2d(32,32);
            new_poses[5]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 128,rand() % 128)- cv::Vec2d(64,64);
            new_poses[0]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 256,rand() % 256)- cv::Vec2d(128,128);
            new_poses[6]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % 512,rand() % 512)- cv::Vec2d(256,256);

            p = dst_im.rowRange(i,i+PATCH_SIZE).colRange(j,j+PATCH_SIZE);

            for(auto & pos: new_poses) {
                //这里需要判断一下位置是不是合理
                if(isnan(pos[0]) || pos[0]<0 || pos[0] > src_im.rows - PATCH_SIZE || pos[1]<0 || pos[1] > src_im.cols - PATCH_SIZE) continue;

                q = src_im.rowRange(pos[0],pos[0]+PATCH_SIZE).colRange(pos[1],pos[1]+PATCH_SIZE);
                new_d = patchDistance(p,q);
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
    cv::Mat mask;
    cv::Mat coord_dist = getCoordDist();
    cv::Mat src_tmp;
    cv::Mat mask_patch;
    cv::Mat src_patch;
    cv::Mat dst_patch;
    cv::Mat aim_patch;

    int new_rows = rows * scale;
    int new_cols = cols * scale;
    int new_patch_size = PATCH_SIZE * scale;
    int new_stride = STRIDE * scale;

    mask = cv::Mat::zeros(new_rows, new_cols, CV_64FC3);
    patched_im = cv::Mat::zeros(new_rows, new_cols, CV_64FC3);

    for(int i = 0; i < new_rows - new_patch_size; i += new_stride) {
        for(int j = 0; j < new_cols - new_patch_size; j += new_stride) {

            aim_patch = patched_im.rowRange(i,i+new_patch_size).colRange(j,j+new_patch_size);

            cv::Vec2d pos = coord.at<cv::Vec2d>(i/scale,j/scale)  *  scale;
            src_patch = src.rowRange(pos[0],pos[0] + new_patch_size).colRange(pos[1],pos[1]+new_patch_size);
            dst_patch = dst.rowRange(i,i + new_patch_size).colRange(j,j+new_patch_size);

            src_patch.convertTo(src_tmp,CV_64FC3);

            cv::add(aim_patch,src_tmp,aim_patch);
            mask_patch = mask.rowRange(i,i+new_patch_size).colRange(j,j+new_patch_size);
            cv::add(cv::Scalar(1,1,1),mask_patch,mask_patch);
        }
    }
    cv::divide(patched_im,mask,patched_im);
    patched_im.convertTo(patched_im,CV_8UC3);
}

double PatchMatch::patchDistance(const cv::Mat &q, const cv::Mat &p) {
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

cv::Mat PatchMatch::getCoord() {
    return coord;
}

cv::Mat PatchMatch::getPatched() {
    generatePatched();
    return patched_im;
}

cv::Mat PatchMatch::getCoordDist() {
    cv::Mat lap;
    cv::Laplacian(coord,lap,CV_64F);
    std::vector<cv::Mat> s;
    cv::split(lap,s);
    cv::magnitude( s[0], s[1],lap);
    cv::normalize(lap,lap,0,1,cv::NORM_MINMAX);
    return lap;
}