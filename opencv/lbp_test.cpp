#include  <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/core/mat.hpp"

// #include<opencv2/legacy/legacy.hpp>
#include<vector>

using namespace std;
using namespace cv;


template <typename _tp>
void getRotationInvariantLBPFeature_(InputArray _src,OutputArray _dst,int radius,int neighbors)
{
    Mat src = _src.getMat();
    //LBP特征图像的行数和列数的计算要准确
    _dst.create(src.rows-2*radius,src.cols-2*radius,CV_8UC1);
    Mat dst = _dst.getMat();
    dst.setTo(0);
    for(int k=0;k<neighbors;k++)
    {
        //计算采样点对于中心点坐标的偏移量rx，ry
        float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));
        //为双线性插值做准备
        //对采样点偏移量分别进行上下取整
        int x1 = static_cast<int>(floor(rx));
        int x2 = static_cast<int>(ceil(rx));
        int y1 = static_cast<int>(floor(ry));
        int y2 = static_cast<int>(ceil(ry));
        //将坐标偏移量映射到0-1之间
        float tx = rx - x1;
        float ty = ry - y1;
        //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        float w1 = (1-tx) * (1-ty);
        float w2 =    tx  * (1-ty);
        float w3 = (1-tx) *    ty;
        float w4 =    tx  *    ty;
        //循环处理每个像素
        for(int i=radius;i<src.rows-radius;i++)
        {
            for(int j=radius;j<src.cols-radius;j++)
            {
                //获得中心像素点的灰度值
                _tp center = src.at<_tp>(i,j);
                //根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = src.at<_tp>(i+x1,j+y1) * w1 + src.at<_tp>(i+x1,j+y2) *w2 \
                    + src.at<_tp>(i+x2,j+y1) * w3 +src.at<_tp>(i+x2,j+y2) *w4;
                //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst.at<uchar>(i-radius,j-radius) |= (neighbor>center) <<(neighbors-k-1);
            }
        }
    }
    //进行旋转不变处理
    for(int i=0;i<dst.rows;i++)
    {
        for(int j=0;j<dst.cols;j++)
        {
            unsigned char currentValue = dst.at<uchar>(i,j);
            unsigned char minValue = currentValue;
            for(int k=1;k<neighbors;k++)
            {
    //循环左移
                unsigned char temp = (currentValue>>(neighbors-k)) | (currentValue<<k);
                if(temp < minValue)
                {
                    minValue = temp;
                }
            }
            dst.at<uchar>(i,j) = minValue;
        }
    }
}



 
int main( )
{
    //从文件中读入图像
    //Mat img = imread("xujiequn1.jpg");
	Mat res;
	Mat img2=imread("test2.png");
	cvtColor(img2, img2, cv::COLOR_BGR2GRAY);
	imshow("org ",img2);  
	getRotationInvariantLBPFeature_<unsigned char>(img2,res,3,8);
	imshow("org->lbp ",res);


	Mat dst;
	Point center(img2.cols/2,img2.rows/2); 
	double angle = 30.0; 
	double scale = 1.0; 
	Mat rotMat = getRotationMatrix2D(center,angle,scale);
	warpAffine(img2,dst,rotMat,img2.size());
	imshow("rotation",dst);

	getRotationInvariantLBPFeature_<unsigned char>(dst,res,3,8);
	imshow("rotation->lbp ",res);

	 waitKey();
}
