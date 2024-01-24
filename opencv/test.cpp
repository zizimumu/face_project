#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>


using namespace cv;

int main(int argc, char *argv[])
{ 
 //   Mat image = Mat::zeros(100, 100, CV_8UC3);
 //   imshow("image", image);
 //   waitKey(10);

	Mat img =imread("test1.jpg");

	//Mat img2(Size(320,240),CV_8UC3);
	//roi 是表示 img 中 Rect(10,10,100,100)区域的对象
	Mat roi(img, Rect(100,100,100,100));


	imshow("image",img);
	imshow("image2",roi);

	waitKey();
	return 0;
 }

