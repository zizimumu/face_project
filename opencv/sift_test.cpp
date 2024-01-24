
#include  <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/xfeatures2d.hpp"

// #include<opencv2/legacy/legacy.hpp>
#include<vector>

using namespace std;
using namespace cv;
 
int main( )
{
    //从文件中读入图像
    Mat img = imread("xujiequn1.jpg");
    Mat img2=imread("test2.png");
 

    //显示图像
    imshow("image before", img);
    imshow("image2 before",img2);
     
 
    //sift特征检测
    // cv::xfeatures2d::SiftFeatureDetector  siftdtc;
    Ptr<Feature2D> siftdtc = xfeatures2d::SIFT::create();
    Ptr<Feature2D> surfdtc = xfeatures2d::SURF::create();


    vector<KeyPoint>kp1,kp2;
 
    siftdtc->detect(img,kp1);

    Mat outimg1;
    drawKeypoints(img,kp1,outimg1);
    imshow("image1 keypoints",outimg1);

    KeyPoint kp;
 
    vector<KeyPoint>::iterator itvc;
//    for(itvc=kp1.begin();itvc!=kp1.end();itvc++)
//    {
 //       cout<<"angle:"<<itvc->angle<<"\t"<<itvc->class_id<<"\t"<<itvc->octave<<"\t"<<itvc->pt<<"\t"<<itvc->response<<endl;
//    }
 
    siftdtc->detect(img2,kp2);
    Mat outimg2;
    drawKeypoints(img2,kp2,outimg2);
    imshow("image2 keypoints",outimg2);
 


surfdtc->detect(img,kp1);
drawKeypoints(img,kp1,outimg1);
 imshow("image1 SURF keypoints",outimg1);
 

/*
    SiftDescriptorExtractor extractor;
    Mat descriptor1,descriptor2;
    BruteForceMatcher<L2<float>> matcher;
    vector<DMatch> matches;
    Mat img_matches;
    extractor.compute(img,kp1,descriptor1);
    extractor.compute(img2,kp2,descriptor2);
 
 
    imshow("desc",descriptor1);
    cout<<endl<<descriptor1<<endl;
    matcher.match(descriptor1,descriptor2,matches);
 
    drawMatches(img,kp1,img2,kp2,matches,img_matches);
    imshow("matches",img_matches);
 */
    //此函数等待按键，按键盘任意键就返回
    waitKey();
    return 0;
}
