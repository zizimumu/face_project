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

#define MAX_S(a,b) {(a) >= (b)?(a):(b)}
#define MIN_S(a,b) {(b) >= (a)?(a):(b)}


using namespace seeta;
using namespace cv;
using namespace std;

struct timeval start_time, stop_time;
std::vector<cv::Point2f> p1s,p2s;

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
#define PIC_SIZE 112


float get_distance(vector<float> &src1,vector<float> &src2)
{
    float res = 0;
    printf("input vector size %d\n",src1.size());
    
    for(int i = 0; i< src1.size();i++){
        res += ((src1[i]-src2[i])*(src1[i]-src2[i]));
    }
    return res;
}
// BGR2RGB
void color_mat2vector(Mat src,vector<float> &dst)
{  
#if 1
    int cnt = 0; 
    if(src.channels() != 3){
        printf("color_mat2vector: not RGB pic\n");
        return;
    }
        

    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++) {
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][2] - 127.5) / 128; 
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][1]- 127.5) / 128;
            dst[cnt++] = (src.ptr<Vec3b>(i)[j][0]- 127.5) / 128;
        }
    }
#else
    for (int i = 0; i < src.rows*src.cols*3; i++){
         dst[i] = 100.0; 
    }

#endif

}

void get_input(char *path, vector<float> &input_array,FaceDetection &detector,FaceAlignment &point_detector)
{
    static int count = 0;
    char name[32];
#if 1
    std::vector<cv::Point2f> p2s;

    cv::Mat color_img = cv::imread(path);  
    cv::Mat gallery_img_gray;

    gettimeofday(&start_time, nullptr); 
    cvtColor(color_img, gallery_img_gray, COLOR_BGR2GRAY);
    //cv::Mat gallery_img_gray = cv::imread(path, cv::IMREAD_GRAYSCALE);  
    gettimeofday(&stop_time, nullptr); 
    
    std::cout << "\ngray time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";

    gettimeofday(&start_time, nullptr); 
    ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
    gallery_img_data_gray.data = gallery_img_gray.data;

    
    std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(gallery_img_data_gray);
    gettimeofday(&stop_time, nullptr); 


    std::cout << "detect time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";
  
  
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
    if (gallery_face_num == 0)
    {
    std::cout << "Faces are not detected.";
    return ;
    }  

    std::cout << "found face :"<<gallery_face_num<<"\n";


    seeta::FacialLandmark gallery_points[5];

    gettimeofday(&start_time, nullptr); 
    point_detector.PointDetectLandmarks(gallery_img_data_gray, gallery_faces[0], gallery_points);
    gettimeofday(&stop_time, nullptr); 
    std::cout << "landmark time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";
    
    for(int i=0;i<5;i++){
        p2s.push_back(cv::Point2f(gallery_points[i].x, gallery_points[i].y));
        //printf("x %f,y %f\n",gallery_points[i].x,gallery_points[i].y);
    }

    //estimateAffinePartial2D estimateRigidTransform
	gettimeofday(&start_time, nullptr); 
    cv::Mat t = cv::estimateAffinePartial2D(p2s,p1s);
    gettimeofday(&stop_time, nullptr); 
    std::cout << "estimateAffinePartial2D time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";

	cv::Mat res_img;
	if(!t.data){
		cout<<"estimate matrix found non\n";	


		int margin =  44;
		int bb[4];

		bb[0] = MAX_S(gallery_faces[0].bbox.x-margin/2,0);
		bb[1] = MAX_S(gallery_faces[0].bbox.y-margin/2,0);
		bb[2] = MIN_S(gallery_faces[0].bbox.x+gallery_faces[0].bbox.width + margin/2, color_img.cols);
		bb[3] = MIN_S(gallery_faces[0].bbox.y+gallery_faces[0].bbox.height + margin/2, color_img.rows);

		//printf("bb %d,%d,%d,%d\n",bb[0],bb[1],bb[2],bb[3]);
		//printf("face box %d,%d,%d,%d\n",gallery_faces[0].bbox.x,gallery_faces[0].bbox.y,gallery_faces[0].bbox.width, \
		//	gallery_faces[0].bbox.height);
		Mat roi(color_img, cv::Rect(bb[0],bb[1],bb[2] - bb[0],bb[3]-bb[1]));
		cv::resize(roi, res_img, cv::Size(112, 112));

	}
    else{
        //std::cout << "found matirx\n  "<<t << "\n";
        
        gettimeofday(&start_time, nullptr);
        warpAffine(color_img, res_img, t,cv::Size(PIC_SIZE,PIC_SIZE));
        gettimeofday(&stop_time, nullptr); 
        std::cout << "warpAffine time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";
        //sprintf(name,"result_%d.bmp",count++);
        //cv::imwrite(name, res_img);
	}
    
    gettimeofday(&start_time, nullptr);
    color_mat2vector(res_img,input_array);  
    gettimeofday(&stop_time, nullptr); 
    std::cout << "warpAffine time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";
#else
    cv::Mat color_img = cv::imread(path);  

    cv::Mat input_img ;
    cv::resize(color_img,input_img,cv::Size(112, 112)); 
    
    color_mat2vector(input_img,input_array);
#endif    
}

void print_vector(vector<float> &src)
{

	cout<<"\n";
	for(int i=0;i<src.size();i++){
		if( ((i % 6) == 0) && (i !=0))
			cout<<"\n";	

		printf(" %8f",src[i]);

	}
	cout<<"\n";
}


int main(int argc, char** argv)
{
  // Initialize face detection model
    int i,size;
    char *path, *path2;
  
  
    if (argc < 3)
        std::cout << "param err";

    gettimeofday(&start_time, nullptr); 

	size = PIC_SIZE*PIC_SIZE*3;
    
    std::vector<float> input_array(size);

    p1s.push_back(cv::Point2f( 38.2946, 51.6963));
    p1s.push_back(cv::Point2f( 73.5318, 51.5014));
    p1s.push_back(cv::Point2f(56.0252, 71.7366));
    p1s.push_back(cv::Point2f( 41.5493, 92.3655));
    p1s.push_back(cv::Point2f( 70.7299, 92.2041));
    
    path = argv[1];
    path2 = argv[2];
    seeta::FaceDetection detector("seeta_fd_frontal_v1.0.bin");
    detector.SetMinFaceSize(80);
    detector.SetScoreThresh(2.f);
    detector.SetImagePyramidScaleFactor(0.8f);
    detector.SetWindowStep(4, 4);

    // Initialize face alignment model 
    seeta::FaceAlignment point_detector("seeta_fa_v1.1.bin");

  
  
    const char* tflite_file = "mobileface_nonquntize_nopre_process.tflite";
    const std::string inputName = "data"; //"data";
    const std::string outputName = "output";

    using TContainer = boost::variant<std::vector<float>>;

    unsigned int outputNumElements = 128;


	using IParser = armnnTfLiteParser::ITfLiteParser;
	auto armnnparser(IParser::Create());
	armnn::INetworkPtr network = armnnparser->CreateNetworkFromBinaryFile(tflite_file);


    // Find the binding points for the input and output nodes 
        
     using BindingPointInfo = armnnTfLiteParser::BindingPointInfo;    
    const std::vector<BindingPointInfo> inputBindings  = { armnnparser->GetNetworkInputBindingInfo(0,inputName) };
    const std::vector<BindingPointInfo> outputBindings = { armnnparser->GetNetworkOutputBindingInfo(0, outputName) };        
 

    std::vector<TContainer> outputDataContainers = { std::vector<float>(outputNumElements)};


    // Optimize the network for a specific runtime compute 
    // device, e.g. CpuAcc, GpuAcc CpuRef
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network,
       {armnn::Compute::CpuAcc, armnn::Compute::CpuRef},  
       runtime->GetDeviceSpec());

       // Load the optimized network onto the runtime device
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optNet));

    

    gettimeofday(&stop_time, nullptr); 
    std::cout << "init time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";

    std::vector<TContainer> inputDataContainers = {std::vector<float>(PIC_SIZE*PIC_SIZE*3)};
    
    
    get_input(path,input_array,detector,point_detector);
    

    // Predict
    gettimeofday(&start_time, nullptr); 
    
    // const std::vector<TContainer> inputDataContainers = {input_array}; 
    inputDataContainers[0] = {input_array}; 
    //runtime->GetProfiler(networkIdentifier)->EnableProfiling(true);

    armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier,
          armnnUtils::MakeInputTensors(inputBindings, inputDataContainers),
          armnnUtils::MakeOutputTensors(outputBindings, outputDataContainers));
  
    //runtime->GetProfiler(networkIdentifier)->Print(std::cout);

    std::vector<float> output1 = boost::get<std::vector<float>>(outputDataContainers[0]);


    
    gettimeofday(&stop_time, nullptr); 
    std::cout << "identify time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";
    
    

#if 1
    get_input(path2,input_array,detector,point_detector);

    gettimeofday(&start_time, nullptr); 
    const std::vector<TContainer> inputDataContainers2 = {input_array}; 
    ret = runtime->EnqueueWorkload(networkIdentifier,
          armnnUtils::MakeInputTensors(inputBindings, inputDataContainers2),
          armnnUtils::MakeOutputTensors(outputBindings, outputDataContainers));

    
    std::vector<float> output2 = boost::get<std::vector<float>>(outputDataContainers[0]);    
    gettimeofday(&stop_time, nullptr); 
    std::cout << "identify time ms: "<< (get_us(stop_time) - get_us(start_time)) / (1000)<<"\n";
    
    printf("distance %f\n",get_distance(output1,output2));


  // cvSaveImage("result.jpg", gallery_img_color);
#endif

  return 0;
}
