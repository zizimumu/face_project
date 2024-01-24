/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "face_detection.h"
#include "face_alignment.h"

using namespace seeta;
using namespace cv;
using namespace std;

#define MAX_S(a,b) {(a) >= (b)?(a):(b)}
#define MIN_S(a,b) {(b) >= (a)?(a):(b)}


std::vector<cv::Point2f> src,dst;
      
int main(int argc, char** argv)
{
  // Initialize face detection model
  int i;
  if (argc < 2)
    std::cout << "param err";
    
  char *path = argv[1];
  seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  // Initialize face alignment model 
  seeta::FaceAlignment point_detector("seeta_fa_v1.1.bin");



  cv::Mat color_img_gray = cv::imread(path);  
	cout<<"img width"<<color_img_gray.cols;
	cout<<"img len"<<color_img_gray.rows;

  cv::Mat gallery_img_gray;
  
  cvtColor(color_img_gray, gallery_img_gray, COLOR_BGR2GRAY);
  //cv::Mat gallery_img_gray = cv::imread(path, cv::IMREAD_GRAYSCALE);  
  
  
  ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
  gallery_img_data_gray.data = gallery_img_gray.data;
  
  std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
  int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
  if (gallery_face_num == 0)
  {
    std::cout << "Faces are not detected.";
    return 0;
  }  
  
  std::cout << "found face :"<<gallery_face_num<<"\n";


  seeta::FacialLandmark gallery_points[5];
  point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
  for ( i = 0; i<5; i++)
  {
    cv::circle(color_img_gray, cv::Point(gallery_points[i].x, gallery_points[i].y), 2,CV_RGB(255, 0, 0),5);

  }



    std::vector<cv::Point2f> p1s,p2s;

    p1s.push_back(cv::Point2f( 38.2946, 51.6963));
    p1s.push_back(cv::Point2f( 73.5318, 51.5014));
    p1s.push_back(cv::Point2f(56.0252, 71.7366));
    p1s.push_back(cv::Point2f( 41.5493, 92.3655));
    p1s.push_back(cv::Point2f( 70.7299, 92.2041));

    for(i=0;i<5;i++){
        p2s.push_back(cv::Point2f(gallery_points[i].x, gallery_points[i].y));
		printf("x %f,y %f\n",gallery_points[i].x,gallery_points[i].y);
	}

    //estimateAffinePartial2D estimateRigidTransform
    cv::Mat t = cv::estimateAffinePartial2D(p2s,p1s);

	cv::Mat res_img;
	if(!t.data){
		cout<<"estimate matrix found non\n";	

		int margin =  44;
		int bb[4];

		bb[0] = MAX_S(gallery_faces[0].bbox.x-margin/2,0);
		bb[1] = MAX_S(gallery_faces[0].bbox.y-margin/2,0);
		bb[2] = MIN_S(gallery_faces[0].bbox.x+gallery_faces[0].bbox.width + margin/2, color_img_gray.cols);
		bb[3] = MIN_S(gallery_faces[0].bbox.y+gallery_faces[0].bbox.height + margin/2, color_img_gray.rows);

		printf("bb %d,%d,%d,%d\n",bb[0],bb[1],bb[2],bb[3]);
		printf("face box %d,%d,%d,%d\n",gallery_faces[0].bbox.x,gallery_faces[0].bbox.y,gallery_faces[0].bbox.width, \
			gallery_faces[0].bbox.height);
		Mat roi(color_img_gray, cv::Rect(bb[0],bb[1],bb[2] - bb[0],bb[3]-bb[1]));
		cv::resize(roi, res_img, cv::Size(112, 112));

	}
	else{
		std::cout << t << "\n";
		warpAffine(color_img_gray, res_img, t,Size(112,112));
	}
    
    
  cv::circle(res_img, cv::Point(38.2946, 51.6963), 2,CV_RGB(0, 255, 0),5);
  cv::circle(res_img, cv::Point(73.5318, 51.5014), 2,CV_RGB(0, 255, 0),5);
  cv::circle(res_img, cv::Point(56.0252, 71.7366), 2,CV_RGB(0, 255, 0),5);
  cv::circle(res_img, cv::Point(41.5493, 92.3655), 2,CV_RGB(0, 255, 0),5);
  cv::circle(res_img, cv::Point(70.7299, 92.2041), 2,CV_RGB(0, 255, 0),5);
  
  
  imshow ("image",color_img_gray);
  imshow ("image_res",res_img);
  
  
  waitKey (0);

  // cvSaveImage("result.jpg", gallery_img_color);


  return 0;
}
