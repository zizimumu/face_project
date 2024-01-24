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
#include <sys/time.h>
#include <fcntl.h>
#include <dirent.h>
#include <pthread.h>
#include <fcntl.h>



#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "face_detection.h"
#include "face_alignment.h"


#include "armnn/BackendId.hpp"
#include "armnn/IRuntime.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#include "TensorIOUtils.hpp"
         
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "boost/variant.hpp"
#include "md5/md5.h"
#include "uvc-gadget.h"
#include "v4l2_test.h"


#define MAX_S(a,b) {(a) >= (b)?(a):(b)}
#define MIN_S(a,b) {(b) >= (a)?(a):(b)}


using namespace seeta;
using namespace cv;
using namespace std;




#define TF_INPUT_PIC_SIZE 112

#define MAX_PATH 128 
#define FEATURE_LEN 128
#define MAX_FACE_NUM 5
#define MAX_INDEX_NUM 1000
#define SPEED_LOOP 10
#define PIC_WIDTH 640
#define PIC_HEIGHT 480

#define V4L2_DEV_PATH "/dev/video0"
#define UVC_DEV_PATH "/dev/video1"

#define FEATRUE_FILE "./feature.bin"
#define SAMPLE_PIC_DIR "./pic"

#define DETECTOR_PATH "./seeta_fd_frontal_v1.0.bin"
#define ALIAGNMENT_PATH "./seeta_fa_v1.1.bin"
#define TF_MODEL_PATH "./mobileface_nonquntize_nopre_process.tflite"



using IParser = armnnTfLiteParser::ITfLiteParser;
using BindingPointInfo = armnnTfLiteParser::BindingPointInfo; 
using TContainer = boost::variant<std::vector<float>>;


struct feature_info{
    char name[128];
    float feature[FEATURE_LEN];
};    
struct ARMNN_P{
    armnn::IRuntimePtr *runtime_pt;
    armnn::NetworkId networkIdentifier;
    std::vector<BindingPointInfo> inputBindings;
    std::vector<BindingPointInfo> outputBindings;
    
    armnn::InputTensors *inputTensors;
    armnn::OutputTensors *outputTensors;
    
    FaceDetection *detector;
    FaceAlignment *point_detector;
    
    md5_state_t md5StateT;
    md5_byte_t md5_res[16];
    
    cv::Mat rgbMat;
    
    unsigned int feature_cnt;
    struct feature_info feature_info[MAX_INDEX_NUM];
};

std::vector<float> input_array(TF_INPUT_PIC_SIZE*TF_INPUT_PIC_SIZE*3);    
std::vector<float> out_array(FEATURE_LEN); 
vector< vector<float> > feature_res(MAX_FACE_NUM, vector<float>(FEATURE_LEN));   
struct timeval start_time, stop_time;
std::vector<cv::Point2f> p1s;
pthread_t ntid;




double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

    
float get_distance(vector<float> &src1,vector<float> &src2)
{
    float res = 0;

    
    for(unsigned int i = 0; i< src1.size();i++){
        res += ((src1[i]-src2[i])*(src1[i]-src2[i]));
    }
    return res;
}

float get_distance_V_B(vector<float> &src_V,float *src_B)
{
    float res = 0;

    //printf("get_distance_V_B input size %d\n",src_V.size());
    
    for(unsigned int i = 0; i< src_V.size();i++){
        res += ((src_V[i]-src_B[i])*(src_V[i]-src_B[i]));
    }
    return res;
}

// BGR2RGB
void mat2vector_bgr2rgb(Mat src,vector<float> &dst)
{  

    int cnt = 0; 
    if(src.channels() != 3){
        printf("mat2vector_bgr2rgb: not RGB pic\n");
        return;
    }
        

    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++) {
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][2] - 127.5) / 128; 
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][1]- 127.5) / 128;
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][0]- 127.5) / 128;
        }
    }
}


void mat2vector_rgb2rgb(Mat src,vector<float> &dst)
{  

    int cnt = 0; 

    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++) {
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][0] - 127.5) / 128; 
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][1]- 127.5) / 128;
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][2]- 127.5) / 128;
        }
    }

}

void vector2buff(vector<float> &src, float *buff, int size)
{  

 

    for (int i = 0; i < size; i++){
        buff[i] = src[i];
    }
    
    //memcpy(buff,&src[0],src.size()*sizeof(float));


}

int  get_input(char *path, vector<float> &input_array,FaceDetection *detector,FaceAlignment *point_detector)
{
    //static int count = 0;
    //char name[32];

    std::vector<cv::Point2f> p2s;

    cv::Mat color_img = cv::imread(path);  
    cv::Mat gallery_img_gray;

    cvtColor(color_img, gallery_img_gray, COLOR_BGR2GRAY);
    //cv::Mat gallery_img_gray = cv::imread(path, cv::IMREAD_GRAYSCALE);  


    ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
    gallery_img_data_gray.data = gallery_img_gray.data;


    std::vector<seeta::FaceInfo> gallery_faces = detector->Detect(gallery_img_data_gray);



  
  
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
    if (gallery_face_num == 0)
    {
        printf("Faces are not detected.\n");
        return -1;
    }  

    printf("found face %d\n",gallery_face_num);


    seeta::FacialLandmark gallery_points[5];

    gettimeofday(&start_time, nullptr); 
    point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
    gettimeofday(&stop_time, nullptr); 

    
    for(int i=0;i<5;i++){
        p2s.push_back(cv::Point2f(gallery_points[i].x, gallery_points[i].y));
    }


    cv::Mat t = cv::estimateAffinePartial2D(p2s,p1s);


	cv::Mat res_img;
	if(!t.data){
		printf("estimate matrix found non\n");	


		int margin =  44;
		int bb[4];

		bb[0] = MAX_S(gallery_faces[0].bbox.x-margin/2,0);
		bb[1] = MAX_S(gallery_faces[0].bbox.y-margin/2,0);
		bb[2] = MIN_S(gallery_faces[0].bbox.x+gallery_faces[0].bbox.width + margin/2, color_img.cols);
		bb[3] = MIN_S(gallery_faces[0].bbox.y+gallery_faces[0].bbox.height + margin/2, color_img.rows);

		Mat roi(color_img, cv::Rect(bb[0],bb[1],bb[2] - bb[0],bb[3]-bb[1]));
		cv::resize(roi, res_img, cv::Size(112, 112));

	}
    else{

        warpAffine(color_img, res_img, t,cv::Size(TF_INPUT_PIC_SIZE,TF_INPUT_PIC_SIZE));
        
	}
    
    
    mat2vector_bgr2rgb(res_img,input_array);   
 
    return 0;
}




int  image_detect(char *path, vector<float> &input_array,ARMNN_P *armnn_pt, int max_num)
{
    //static int count = 0;
    //char name[32];
    int i,cnt = 0;
    cv::Mat res_img;
    cv::Mat gallery_img_gray;

    std::vector<cv::Point2f> p2s(5);

    cv::Mat color_img;

    color_img = cv::imread(path);
    
    if( (color_img.data == NULL) ){
        printf("unsupport image file %s\n",path);
        return -1;
    }
    if( color_img.channels() != 3){
        printf("only support 3 channel image file, current channel %d\n",color_img.channels());
        return -1;        
    }
    

    cvtColor(color_img, gallery_img_gray, COLOR_BGR2GRAY);
    //cv::Mat gallery_img_gray = cv::imread(path, cv::IMREAD_GRAYSCALE);  


    ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
    gallery_img_data_gray.data = gallery_img_gray.data;


    std::vector<seeta::FaceInfo> gallery_faces = armnn_pt->detector->Detect(gallery_img_data_gray);



  
  
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
    if (gallery_face_num == 0)
    {
        //std::cout << "Faces are not detected.";
        return -1;
    }  

    //std::cout << "found face :"<<gallery_face_num<<"\n";


    seeta::FacialLandmark gallery_points[5];

    cnt = MIN_S(gallery_face_num,max_num);
    
    for(i=0;i<gallery_face_num;i++){
    

        armnn_pt->point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[i], gallery_points);

        
        for(int i=0;i<5;i++){
            p2s[i] = (cv::Point2f(gallery_points[i].x, gallery_points[i].y));
        }


        cv::Mat t = cv::estimateAffinePartial2D(p2s,p1s);


        
        if(!t.data){
            printf("estimate matrix found non\n");	


            int margin =  44;
            int bb[4];

            bb[0] = MAX_S(gallery_faces[0].bbox.x-margin/2,0);
            bb[1] = MAX_S(gallery_faces[0].bbox.y-margin/2,0);
            bb[2] = MIN_S(gallery_faces[0].bbox.x+gallery_faces[0].bbox.width + margin/2, color_img.cols);
            bb[3] = MIN_S(gallery_faces[0].bbox.y+gallery_faces[0].bbox.height + margin/2, color_img.rows);

            Mat roi(color_img, cv::Rect(bb[0],bb[1],bb[2] - bb[0],bb[3]-bb[1]));
            cv::resize(roi, res_img, cv::Size(112, 112));

        }
        else{

            warpAffine(color_img, res_img, t,cv::Size(TF_INPUT_PIC_SIZE,TF_INPUT_PIC_SIZE));
            
        }
        
        
        mat2vector_bgr2rgb(res_img,input_array);   

        
  
#if 0 
        inputDataContainers[0] = {input_array};
        armnn::InputTensors inputTensors = armnnUtils::MakeInputTensors(armnn_pt->inputBindings, inputDataContainers);
        armnn::OutputTensors outputTensors = armnnUtils::MakeOutputTensors(armnn_pt->outputBindings, outputDataContainers);
        //armnn::Status ret = 
        (*(armnn_pt->runtime_pt))->EnqueueWorkload(armnn_pt->networkIdentifier,inputTensors,outputTensors);
        
        printf("%d \n",armnn_pt->inputBindings[0].first);
      

        feature_res[i] = boost::get<std::vector<float>>(outputDataContainers[0]);
     
        
#else
    

        
        //armnn::InputTensors inputTensors
        //{
        //    {armnn_pt->inputBindings[0].first, armnn::ConstTensor(armnn_pt->inputBindings[0].second, input_array.data())}
        //};
        //
        //armnn::OutputTensors outputTensors
        //{
        //    {armnn_pt->outputBindings[0].first, armnn::Tensor(armnn_pt->outputBindings[0].second, feature_res[0].data())}
        //};
        //
       

        //armnn::InputTensors inputTensors;
        //armnn::ConstTensor inputTensor(armnn_pt->inputBindings[0].second, input_array.data());
        //inputTensors.push_back(std::make_pair(armnn_pt->inputBindings[0].first, inputTensor));
        //
        //
        //armnn::OutputTensors outputTensors;
        //armnn::Tensor outputTensor(armnn_pt->outputBindings[0].second, feature_res[0].data());
        //outputTensors.push_back(std::make_pair(armnn_pt->outputBindings[0].first, outputTensor));
            
      
        (*(armnn_pt->runtime_pt))->EnqueueWorkload(armnn_pt->networkIdentifier,*(armnn_pt->inputTensors),*(armnn_pt->outputTensors));
        feature_res[i] = out_array;
    
#endif
    
        
    }
    
    
 
    return cnt;
}



int  image_detect_speed1(char *path, vector<float> &input_array,ARMNN_P *armnn_pt, int max_num)
{

    int i,j,cnt = 0;
    cv::Mat res_img;
    cv::Mat gallery_img_gray;
	cv::Mat color_img;
	cv::Mat t;
	int32_t gallery_face_num;
    std::vector<cv::Point2f> p2s(5);

    
	seeta::FacialLandmark gallery_points[5];
	std::vector<seeta::FaceInfo> gallery_faces;

    color_img = cv::imread(path);
    
    if( (color_img.data == NULL) ){
        printf("unsupport image file %s\n",path);
        return -1;
    }
    if( color_img.channels() != 3){
        printf("only support 3 channel image file, current channel %d\n",color_img.channels());
        return -1;        
    }


		gettimeofday(&start_time, nullptr); 
	    cvtColor(color_img, gallery_img_gray, COLOR_BGR2GRAY);
		gettimeofday(&stop_time, nullptr); 
		printf("gray ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );

		gettimeofday(&start_time, nullptr);
	    ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
	    gallery_img_data_gray.data = gallery_img_gray.data;

		
	    gallery_faces = armnn_pt->detector->Detect(gallery_img_data_gray);


		gettimeofday(&stop_time, nullptr);
		printf("detect ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );

	    gallery_face_num = static_cast<int32_t>(gallery_faces.size());
	    if (gallery_face_num == 0){
	        return -1;
	    }  

	    cnt = MIN_S(gallery_face_num,max_num);
	    
	    for(i=0;i<gallery_face_num;i++){
	    	gettimeofday(&start_time, nullptr);
	        armnn_pt->point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[i], gallery_points);
			gettimeofday(&stop_time, nullptr);
			printf("landmark ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );


			
	        for(int i=0;i<5;i++){
	            p2s[i] = (cv::Point2f(gallery_points[i].x, gallery_points[i].y));
	        }

			gettimeofday(&start_time, nullptr);
	        t = cv::estimateAffinePartial2D(p2s,p1s);
						gettimeofday(&stop_time, nullptr);
			printf("estimateAffinePartial2D ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );
	        
	        if(!t.data){
	            printf("estimate matrix found non\n");	

	            int margin =  44;
	            int bb[4];

	            bb[0] = MAX_S(gallery_faces[0].bbox.x-margin/2,0);
	            bb[1] = MAX_S(gallery_faces[0].bbox.y-margin/2,0);
	            bb[2] = MIN_S(gallery_faces[0].bbox.x+gallery_faces[0].bbox.width + margin/2, color_img.cols);
	            bb[3] = MIN_S(gallery_faces[0].bbox.y+gallery_faces[0].bbox.height + margin/2, color_img.rows);

	            Mat roi(color_img, cv::Rect(bb[0],bb[1],bb[2] - bb[0],bb[3]-bb[1]));
	            cv::resize(roi, res_img, cv::Size(112, 112));

	        }
	        else{
				gettimeofday(&start_time, nullptr);
	            warpAffine(color_img, res_img, t,cv::Size(TF_INPUT_PIC_SIZE,TF_INPUT_PIC_SIZE));
						gettimeofday(&stop_time, nullptr);
				printf("warpAffine ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );	            
	        }

			gettimeofday(&start_time, nullptr);
	        mat2vector_bgr2rgb(res_img,input_array);   
			gettimeofday(&stop_time, nullptr);
			printf("mat2vector ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );	  

			gettimeofday(&start_time, nullptr);
	        (*(armnn_pt->runtime_pt))->EnqueueWorkload(armnn_pt->networkIdentifier,*(armnn_pt->inputTensors),*(armnn_pt->outputTensors));
	        feature_res[i] = out_array;
			gettimeofday(&stop_time, nullptr);
			printf("identify ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );	  			
	        
	    }
	//}

   
    return cnt;
}


int  image_detect_speed(char *path, vector<float> &input_array,ARMNN_P *armnn_pt, int max_num)
{

    int i,j,cnt = 0;
    cv::Mat res_img;
    cv::Mat gallery_img_gray;
	cv::Mat color_img;
	cv::Mat t;
	int32_t gallery_face_num;
    std::vector<cv::Point2f> p2s(5);

    
	seeta::FacialLandmark gallery_points[5];
	std::vector<seeta::FaceInfo> gallery_faces;

    color_img = cv::imread(path);
    
    if( (color_img.data == NULL) ){
        printf("unsupport image file %s\n",path);
        return -1;
    }
    if( color_img.channels() != 3){
        printf("only support 3 channel image file, current channel %d\n",color_img.channels());
        return -1;        
    }

	//gettimeofday(&start_time, nullptr);   

	//for(j=0;j<SPEED_LOOP;j++){	

	    cvtColor(color_img, gallery_img_gray, COLOR_BGR2GRAY);


	    ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
	    gallery_img_data_gray.data = gallery_img_gray.data;


	gettimeofday(&start_time, nullptr); 
	for(j=0;j<SPEED_LOOP;j++){	
		
	    gallery_faces = armnn_pt->detector->Detect(gallery_img_data_gray);

	    gallery_face_num = static_cast<int32_t>(gallery_faces.size());
	    if (gallery_face_num == 0){
	        return -1;
	    }  

	    cnt = MIN_S(gallery_face_num,max_num);
	    
	    for(i=0;i<gallery_face_num;i++){
	    
	        armnn_pt->point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[i], gallery_points);
	        for(int i=0;i<5;i++){
	            p2s[i] = (cv::Point2f(gallery_points[i].x, gallery_points[i].y));
	        }
	        t = cv::estimateAffinePartial2D(p2s,p1s);
	        
	        if(!t.data){
	            printf("estimate matrix found non\n");	

	            int margin =  44;
	            int bb[4];

	            bb[0] = MAX_S(gallery_faces[0].bbox.x-margin/2,0);
	            bb[1] = MAX_S(gallery_faces[0].bbox.y-margin/2,0);
	            bb[2] = MIN_S(gallery_faces[0].bbox.x+gallery_faces[0].bbox.width + margin/2, color_img.cols);
	            bb[3] = MIN_S(gallery_faces[0].bbox.y+gallery_faces[0].bbox.height + margin/2, color_img.rows);

	            Mat roi(color_img, cv::Rect(bb[0],bb[1],bb[2] - bb[0],bb[3]-bb[1]));
	            cv::resize(roi, res_img, cv::Size(112, 112));

	        }
	        else{

	            warpAffine(color_img, res_img, t,cv::Size(TF_INPUT_PIC_SIZE,TF_INPUT_PIC_SIZE));
	            
	        }
	 
	        mat2vector_bgr2rgb(res_img,input_array);   
	     
	        (*(armnn_pt->runtime_pt))->EnqueueWorkload(armnn_pt->networkIdentifier,*(armnn_pt->inputTensors),*(armnn_pt->outputTensors));
	        feature_res[i] = out_array;
	        
	    }
	}

	gettimeofday(&stop_time, nullptr);   
    return cnt;
}

int  camera_detect(cv::Mat &color_img, vector<float> &input_array,ARMNN_P *armnn_pt, int max_num)
{
    

    //static int count = 0;
    //char name[32];
    int i,cnt = 0;

    cv::Mat res_img;
    cv::Mat gallery_img_gray;
	

    std::vector<cv::Point2f> p2s(5);


    if( (color_img.data == NULL) ){
        printf("unsupport capture frame \n");
        return -1;
    }
    if( color_img.channels() != 3){
        printf("only support 3 channel frame, current channel %d\n",color_img.channels());
        return -1;        
    }
    

    cvtColor(color_img, gallery_img_gray, COLOR_BGR2GRAY);
    //cv::Mat gallery_img_gray = cv::imread(path, cv::IMREAD_GRAYSCALE);  


    ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
    gallery_img_data_gray.data = gallery_img_gray.data;


    std::vector<seeta::FaceInfo> gallery_faces = armnn_pt->detector->Detect(gallery_img_data_gray);



  
  
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
    if (gallery_face_num == 0)
    {
        //std::cout << "Faces are not detected.";
        return -1;
    }  

    //std::cout << "found face :"<<gallery_face_num<<"\n";


    seeta::FacialLandmark gallery_points[5];

    cnt = MIN_S(gallery_face_num,max_num);
    
    for(i=0;i<gallery_face_num;i++){
    

        armnn_pt->point_detector->PointDetectLandmarks(gallery_img_data_gray, gallery_faces[i], gallery_points);

        
        for(int i=0;i<5;i++){
            p2s[i] = (cv::Point2f(gallery_points[i].x, gallery_points[i].y));
        }


        cv::Mat t = cv::estimateAffinePartial2D(p2s,p1s);


        
        if(!t.data){
            printf("estimate matrix found non\n");


            int margin =  44;
            int bb[4];

            bb[0] = MAX_S(gallery_faces[0].bbox.x-margin/2,0);
            bb[1] = MAX_S(gallery_faces[0].bbox.y-margin/2,0);
            bb[2] = MIN_S(gallery_faces[0].bbox.x+gallery_faces[0].bbox.width + margin/2, color_img.cols);
            bb[3] = MIN_S(gallery_faces[0].bbox.y+gallery_faces[0].bbox.height + margin/2, color_img.rows);

            Mat roi(color_img, cv::Rect(bb[0],bb[1],bb[2] - bb[0],bb[3]-bb[1]));
            cv::resize(roi, res_img, cv::Size(112, 112));

        }
        else{

            warpAffine(color_img, res_img, t,cv::Size(TF_INPUT_PIC_SIZE,TF_INPUT_PIC_SIZE));
            
        }
        
        
        mat2vector_rgb2rgb(res_img,input_array);  

        (*(armnn_pt->runtime_pt))->EnqueueWorkload(armnn_pt->networkIdentifier,*(armnn_pt->inputTensors),*(armnn_pt->outputTensors));
        feature_res[i] = out_array;
    
    }

 
    return cnt;
}



int dirwalk_build_feature(char *dir,ARMNN_P *armnn_pt)
{
	char name[256];
	struct dirent *dp;
	DIR *dfd;

	
	if((dfd = opendir(dir)) == NULL){
		fprintf(stderr, "dirwalk: can't open %s\n", dir);
		return -1;
	}
	
	while((dp = readdir(dfd)) != NULL){ //读目录记录项
		if(strcmp(dp->d_name, ".") == 0 || strcmp(dp -> d_name, "..") == 0){
			continue;  //跳过当前目录以及父目录
		}
		

		if( dp->d_type == DT_DIR){
			printf("dir skip: %s\n",dp->d_name);
			continue;

		}
		if( dp->d_type == DT_REG){
			if(strlen(dp -> d_name) > MAX_PATH){
				printf("file name %s too long， skip\n",dp->d_name);
				continue;
			}
			else{
                sprintf(name, "%s/%s", dir, dp->d_name);
                
                printf("reading %s\n",dp->d_name);
                if(image_detect(name, input_array, armnn_pt, 1) == 1){
                    
                    memcpy(armnn_pt->feature_info[armnn_pt->feature_cnt].name,dp->d_name,strlen(dp->d_name));
                    vector2buff(feature_res[0],armnn_pt->feature_info[armnn_pt->feature_cnt].feature,FEATURE_LEN);
                    armnn_pt->feature_cnt++;
                }
                else{
                    printf("find no face in %s\n",dp->d_name);
                }
                    
			}
		}


	}
	
    printf("find %d featrue picture\n",armnn_pt->feature_cnt);
	closedir(dfd);
    
    return armnn_pt->feature_cnt;
}



void print_vector(vector<float> &src)
{

	for(unsigned int i=0;i<src.size();i++){
		if( ((i % 6) == 0) && (i !=0))
			printf("\n");	

		printf(" %8f",src[i]);

	}
	printf("\n");
}

int check_md5(FILE *file,ARMNN_P *armnn_pt)
{
    unsigned int size;
    unsigned char *buf = NULL;
    unsigned char *src_md5 = NULL;
    int ret = 0;
    unsigned int md_sz,i;
    char md5String[33] = { '\0' };
    
    

    fseek(file, 0, SEEK_END);
    size = ftell(file);
    rewind(file);
    
    printf("feartue file size %d\n",size);


    md_sz = sizeof(armnn_pt->md5_res);
    
    if(size <= md_sz || size > (1024*1024*100)){
        printf("feature file error");
        return -1;
    }

    buf = new unsigned char[size];
    if(buf == NULL){
        printf("new buf error\n");
        return -1;
    }

            
    if(fread(buf,size, 1, file) == 1){
        
        md5_init(&armnn_pt->md5StateT);
        md5_append(&armnn_pt->md5StateT,buf,size - md_sz);
        md5_finish(&armnn_pt->md5StateT, armnn_pt->md5_res);

        src_md5 = buf + size - md_sz;
        for(i=0;i<md_sz;i++){
            if(armnn_pt->md5_res[i] != src_md5[i]){
                printf("md5 check error\n");
                ret =  -1;
                
                
                for(int i = 0;i<16;i++)
                {
                    snprintf(md5String+i*2,3,"%02X",armnn_pt->md5_res[i]);
                }
                md5String[32] = '\0';
                printf("caculate md5 is %s\n",md5String);

                for(int i = 0;i<16;i++)
                {
                    snprintf(md5String+i*2,3,"%02X",src_md5[i]);
                }
                md5String[32] = '\0';
                printf("cource file md5 is %s\n",md5String);
                
        
                break;
            }
                
        }
    }
    else{
        printf("read file error\n");
        ret = -1;
    }
        
    
    delete[] buf;
    
    return ret;
}


int create_feature_file(ARMNN_P *armnn_pt)
{
    FILE *file;
    unsigned int sz;
    unsigned char *buf = NULL;
    char md5String[33] = { '\0' };
    
    if(dirwalk_build_feature(SAMPLE_PIC_DIR,armnn_pt) >0 ){
        //file = fopen(FEATRUE_FILE, "wbx")
        
        sz = armnn_pt->feature_cnt*sizeof(feature_info);
        
        md5_init(&armnn_pt->md5StateT);
        md5_append(&armnn_pt->md5StateT,(unsigned char *)armnn_pt->feature_info,sz);
        md5_finish(&armnn_pt->md5StateT, armnn_pt->md5_res);
        
        
        file = fopen(FEATRUE_FILE, "wb");
        if(file == NULL){
            printf("open file %s error\n",FEATRUE_FILE);
            return -1;
        }

        
        buf = new unsigned char[sz+sizeof(armnn_pt->md5_res)];
        if(buf == NULL){
            printf("new buf error\n");
            return -1;
        }
        

        
        
        memcpy(buf,(unsigned char *)armnn_pt->feature_info,sz);
        memcpy(buf+sz,(unsigned char *)armnn_pt->md5_res,sizeof(armnn_pt->md5_res));
        
        fwrite(buf,sz+sizeof(armnn_pt->md5_res), 1, file);

        fclose(file);
        delete[] buf;

        
//        for(int i = 0;i<16;i++)
//        {
//            snprintf(md5String+i*2,3,"%02X",armnn_pt->md5_res[i]);
//        }
//        md5String[32] = '\0';
//        printf("the file md5 is %s\n",md5String);

    
        printf("write feature OK\n");
    }
    return 0;

}

int check_featrue_file(ARMNN_P *armnn_pt)
{
    FILE *file;
    unsigned int size;
    //unsigned int sz;
    //unsigned char *buf = NULL;
    
     if ( file = fopen(FEATRUE_FILE, "r")) {

        printf("find feature file\n");
        if(check_md5(file,armnn_pt)){
            printf("something wrong in feature file, try to read picture to rebuild\n");
            create_feature_file(armnn_pt);
        }
        else{
        
            fseek(file, 0, SEEK_END);
            size = ftell(file);
            rewind(file);
        
        
            if(fread((unsigned char *)armnn_pt->feature_info,size - sizeof(armnn_pt->md5_res), 1, file) == 1){
                armnn_pt->feature_cnt = ( size - sizeof(armnn_pt->md5_res) ) / sizeof(feature_info);
                
                printf("find total %d features in feature file\n",armnn_pt->feature_cnt);
            }
        }
        
        fclose(file);

    } else {//find no file
    
        printf("find no feature file, try to read picture to rebuild\n");
        create_feature_file(armnn_pt);

    }  
    return 0;
    
}

int find_best_compatible(vector<float> &src, ARMNN_P *armnn_pt, float *score)
{
    float mini = 16777216, dis = 0;
    int ret = -1;
    
    for(unsigned int i = 0;i< armnn_pt->feature_cnt;i++){
        dis = get_distance_V_B(src,armnn_pt->feature_info[i].feature);
        
        // printf("distance %f\n",dis);
        if(dis < mini){
            mini = dis;
            ret = i;
            
        }
    }
    
    *score = mini;
    
    return ret;
}



void *thread_uvc(void *arg)
{
	unsigned int cnt = 0;
	int fd;

	printf("thread_uvc running\n");

	uvc_event_loop();
	return((void *)0);
}


extern struct V4L2_DEV *open_camera(char *path, unsigned int cam_width,  unsigned int cam_height, int uvc_en);
extern int get_capture_rgb(cv::Mat &rgbMat, struct V4L2_DEV *v4l2);


static void usage(const char *argv0)
{
    printf("Usage: %s [options]\n", argv0);
    printf("Available options are\n");
    printf(" -u		uvc device path\n");
    printf( "-v		v4l2 capture device path\n");
	printf( "-i		only use image as input\n");
	printf( "-c		only use camera as input\n");
	printf( "-o		enable uvc gadget funtion\n");
	printf( "-p		UVC max packet size,must be equal to kernel driver\n");


}

int main(int argc, char** argv)
{
  // Initialize face detection model
    int size,i,index;
     int ret,opt;
    char *image_path = NULL, *uvc_path = UVC_DEV_PATH,*v4l2_path = V4L2_DEV_PATH;
    float score;
	int camera_en = 0;
	int uvc_en = 0;
	unsigned int uvcMaxPack = 512;
	int fd_v4l2;
	struct V4L2_DEV *v4l2 = NULL;
	unsigned int cnt = 0;



    while ((opt = getopt(argc, argv, "u:v:i:cp:o")) != -1) {
        switch (opt) {
        case 'u':
            uvc_path = optarg;
            break;

        case 'v':
            v4l2_path = optarg;
            break;

        case 'i':
            image_path = optarg;
            break;
        case 'o':
            uvc_en = 1;
			printf("uvc gadget enable\n");
            break;

        case 'c':
			camera_en = 1;
			printf("camera mode enable\n");
            break;
        case 'p':
			uvcMaxPack = atoi(optarg);;
            break;

        default:
            printf("Invalid option '-%c'\n", opt);
            usage(argv[0]);
            return 1;
        }
    }



	if(!camera_en && image_path == NULL){
		printf("must input a image path or enable camera\n");
		return 0;
	}


    gettimeofday(&start_time, nullptr); 


	v4l2 = open_camera(v4l2_path,PIC_WIDTH,PIC_HEIGHT,uvc_en);
	if((v4l2 == NULL)){
        printf("open camera failed\n");
        return -1;
    }

	if(camera_en && uvc_en){
		if(!uvc_init(uvc_path,uvcMaxPack,v4l2)){
			ret = pthread_create(&ntid, NULL, thread_uvc, NULL); 
			if (ret != 0){ 
				printf("can't create thread_uvc\n");	
			}
		}
		else{
			printf("uvc open failed\n");
		}
	}


	ARMNN_P *armnn_pt = new ARMNN_P;
	size = TF_INPUT_PIC_SIZE*TF_INPUT_PIC_SIZE*3;
    memset(armnn_pt,0,sizeof(ARMNN_P));


    p1s.push_back(cv::Point2f( 38.2946, 51.6963));
    p1s.push_back(cv::Point2f( 73.5318, 51.5014));
    p1s.push_back(cv::Point2f(56.0252, 71.7366));
    p1s.push_back(cv::Point2f( 41.5493, 92.3655));
    p1s.push_back(cv::Point2f( 70.7299, 92.2041));
    

    seeta::FaceDetection detector(DETECTOR_PATH);
    detector.SetMinFaceSize(80);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);

    // Initialize face alignment model 
    seeta::FaceAlignment point_detector(ALIAGNMENT_PATH);

  
  
    const std::string inputName = "data"; 
    const std::string outputName = "output";


	
	auto armnnparser(IParser::Create());
	armnn::INetworkPtr network = armnnparser->CreateNetworkFromBinaryFile(TF_MODEL_PATH);


    armnn_pt->inputBindings.push_back(armnnparser->GetNetworkInputBindingInfo(0,inputName));
    armnn_pt->outputBindings.push_back(armnnparser->GetNetworkOutputBindingInfo(0, outputName)); 

    // Optimize the network for a specific runtime compute 
    // device, e.g. CpuAcc, GpuAcc CpuRef
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network,
       {armnn::Compute::CpuAcc, armnn::Compute::CpuRef},  
       runtime->GetDeviceSpec());

       // Load the optimized network onto the runtime device
    
    runtime->LoadNetwork(armnn_pt->networkIdentifier, std::move(optNet));

    
    armnn::InputTensors inputTensors_{
        {armnn_pt->inputBindings[0].first, armnn::ConstTensor(armnn_pt->inputBindings[0].second, input_array.data())}
    };

    armnn::OutputTensors outputTensors_{
        {armnn_pt->outputBindings[0].first, armnn::Tensor(armnn_pt->outputBindings[0].second, out_array.data())}
    };
    
    
    armnn_pt->detector = &detector;
    armnn_pt->point_detector = &point_detector;
    armnn_pt->runtime_pt = &runtime;
    armnn_pt->inputTensors = &inputTensors_;
    armnn_pt->outputTensors = &outputTensors_ ;
    
    check_featrue_file(armnn_pt);
    
    gettimeofday(&stop_time, nullptr); 


    printf("init time ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );


    if(image_path){
		for(int dd = 0;dd <3;dd++)
			if(image_detect_speed1(image_path, input_array, armnn_pt, 1) == 1){
				//printf("face identify time ms: %d\n",(int)(get_us(stop_time) - get_us(start_time)) / (1000) );

	            index = find_best_compatible(feature_res[0],armnn_pt,&score);
	            if(index >=0){
	                printf("find best compatible %s, score %f\n",armnn_pt->feature_info[index].name,score);
	            }		
				//return 0;
			}
	}
	else{


		struct timeval time1,time2,time3;
	    while(1){

			//printf("main go to sleep\n");
			//while(1)
			//	sleep(1);

			gettimeofday(&start_time, nullptr); 
	        get_capture_rgb(armnn_pt->rgbMat,v4l2);
			gettimeofday(&time1, nullptr); 
			
	        ret = camera_detect(armnn_pt->rgbMat, input_array, armnn_pt, MAX_FACE_NUM);
			gettimeofday(&time2, nullptr); 
			
	        if(ret > 0){



				if(ret >1)
					printf("multi face detected\n");
	            for(i=0;i<ret;i++){
	                //printf("get feature successfully\n");
	                index = find_best_compatible(feature_res[i],armnn_pt,&score);
	                if(index >=0){

						gettimeofday(&time3, nullptr); 
						printf("\ncapture ms: %d\n",(int)(get_us(time1) - get_us(start_time)) / (1000) );
						printf("detect  ms: %d\n",(int)(get_us(time2) - get_us(time1)) / (1000) );
						printf("compare ms: %d\n",(int)(get_us(time3) - get_us(time2)) / (1000) );


						
	                    printf("find best compatible %s, score %f\n",armnn_pt->feature_info[index].name,score);
	                }
	            }
			
	        }    
	    }

	}
   
#if 0
    image_detect(path, input_array, armnn_pt, 1);
    std::vector<float> output1 = feature_res[0];

    image_detect(path2, input_array, armnn_pt, 1);
    std::vector<float> output2 = feature_res[0];
    
    printf("distance %f\n",get_distance(output1,output2));

    //if(argc > 1){
    //    printf("reading file %s\n",path);
    //    if(image_detect(path, input_array, armnn_pt, 1) == 1){
    //        printf("get feature successfully\n");
    //        int ret = find_best_compatible(feature_res[0],armnn_pt,&score);
    //        if(ret >=0){
    //            printf("find best compatible %s\n",armnn_pt->feature_info[ret].name);
    //        }
    //    }
    //}

    while(1){
        get_capture_rgb(armnn_pt->rgbMat);
        ret = camera_detect(armnn_pt->rgbMat, input_array, armnn_pt, MAX_FACE_NUM);
        if(ret > 0){
            for(i=0;i<ret;i++){
                //printf("get feature successfully\n");
                index = find_best_compatible(feature_res[i],armnn_pt,&score);
                if(index >=0){
                    printf("find best compatible %s, score %f\n",armnn_pt->feature_info[index].name,score);
                }
            }
        }    
    }
    printf("done\n");
    
 #endif  


  return 0;
}
