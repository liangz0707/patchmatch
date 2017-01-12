//
// Created by LZMAC on 2017/1/10.
//

#include "LocalPatchMatch.h"
LocalPatchMatch::LocalPatchMatch(){}
LocalPatchMatch::LocalPatchMatch(cv::Mat src, cv::Mat dst, double scale, int local_win):PatchMatch(src,dst),local_win(local_win){}
LocalPatchMatch::LocalPatchMatch(cv::Mat src, cv::Mat dst,cv::Mat src_im, cv::Mat dst_im, double scale, int local_win):PatchMatch(src,dst,src_im,dst_im),local_win(local_win){}

/*
 * 局部初始化的方式就是直接使用限定区域的patch尽心初始化
 */
void LocalPatchMatch::init() {
    std::cout<<"new init"<<std::endl;
    cols = dst_im.cols;
    rows = dst_im.rows;
    diff = cv::Mat(rows,cols,CV_64FC1);
    coord = cv::Mat(rows,cols,CV_64FC2);
    patched_im = cv::Mat::zeros(rows,cols,CV_8UC3);
    srand( (unsigned)time( NULL ) );
    int x,y;

    for(int i = 0; i < rows - PATCH_SIZE; i+=STRIDE) {
        for (int j = 0; j < cols - PATCH_SIZE; j+=STRIDE) {


            x = i + rand() % (local_win) - local_win / 2;
            y = j + rand() % (local_win) - local_win / 2;
            x = std::max(x,0);
            y = std::max(y,0);
            x = std::min(src_im.rows-PATCH_SIZE,x);
            y = std::min(src_im.cols-PATCH_SIZE,y);

            coord.at<cv::Vec2d>(i, j) = cv::Vec2d(i,j);
            cv::Vec2d pos = coord.at<cv::Vec2d>(i, j);

            cv::Mat p = dst_im.rowRange(i,i+PATCH_SIZE).colRange(j,j+PATCH_SIZE);
            cv::Mat q = src_im.rowRange(pos[0],pos[0]+PATCH_SIZE).colRange(pos[1],pos[1]+PATCH_SIZE);

            diff.at<double>(i, j) = patchDistance(p,q);
        }
    }
    unchanged = false;
}

void LocalPatchMatch::patchmatch() {
    std::cout<<"new patch match"<<std::endl;
    int itr = 0;
    while( itr ++ < MAX_ITR){
        unchanged = true;
        time_t a= clock();
        neighborPropgation();
        localPropgation();
        time_t b = clock();
    }
}



void LocalPatchMatch::localPropgation() {
    cv::Mat p;
    cv::Mat q;
    double new_d;
    std::vector<cv::Vec2d> new_poses;
    new_poses.reserve(7);
    new_poses.resize(7);
    int pwin = local_win / 2;
    for(int i = 0; i < rows - PATCH_SIZE; i += STRIDE){
        for (int j = 0; j < cols - PATCH_SIZE; j += STRIDE) {

            new_poses[1]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % local_win,rand() % local_win)- cv::Vec2d(pwin,pwin);
            new_poses[2]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % local_win,rand() % local_win)- cv::Vec2d(pwin,pwin);
            new_poses[3]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % local_win,rand() % local_win)- cv::Vec2d(pwin,pwin);
            new_poses[4]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % local_win,rand() % local_win)- cv::Vec2d(pwin,pwin);
            new_poses[5]=coord.at<cv::Vec2d>(i, j) + cv::Vec2d(rand() % local_win,rand() % local_win)- cv::Vec2d(pwin,pwin);

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

cv::Mat LocalPatchMatch::drawCoord(){
    int STEP = 20;
    int PAD = 20;
    cv::Mat coorddir= src.clone();//(coord.rows, coord.cols, CV_64FC1);
    //coorddir.setTo(cv::Scalar(0,0,0));
    for(int i = PAD; i < coord.rows - PATCH_SIZE; i+= STEP) {
        for (int j = PAD; j < coord.cols - PATCH_SIZE; j+=STEP) {
            cv::Point2d start(j,i);
            cv::Point2d end;
            end.x =coord.at<cv::Vec2d>(i,j)[1];
            end.y =coord.at<cv::Vec2d>(i,j)[0];
            std::cout<<start<<"->"<<end<<":"<<diff.at<double>(i,j)<<std::endl;
            cv::line(coorddir, start, end, cv::Scalar(0,255,0));
        }
    }
    return coorddir;
}