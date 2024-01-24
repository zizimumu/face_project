#include <stdio.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <string.h>  
#include <fcntl.h>             
#include <unistd.h>
#include <sys/mman.h> 
#include<time.h>
#include <linux/videodev2.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "v4l2_test.h"
struct buffer{  
	void *start;  
	unsigned int length;  
}; 

#define BUFF_COUNT 1
#define CAM_PIX_FMT V4L2_PIX_FMT_YUYV
#define  VIDEO_WIDTH  640
#define  VIDEO_HEIGHT  480
 //0/9 


using namespace cv;

struct buffer buffers[BUFF_COUNT];
struct v4l2_buffer buf[BUFF_COUNT]; 
cv::Mat yuvMat;    


struct V4L2_DEV v4l2_dev;

struct V4L2_DEV *open_camera(char *path, unsigned int cam_width,  unsigned int cam_height, int uvc_en)
{  

	//1.open device.打开摄像头设备 
	int index = -1;

    struct v4l2_capability cap;
    struct v4l2_fmtdesc fmtdesc;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req; 
    enum v4l2_buf_type type; 
    unsigned int i = 0,w,h; 
    int fd;
	unsigned char *pt;
	
    

	memset(&v4l2_dev,0,sizeof(v4l2_dev));
	
	fd = open(path,O_RDWR,0);
	if(fd<0){
		printf("open %s device failed.\n",path);
        return NULL;
	}
 

	
	if(ioctl(fd,VIDIOC_QUERYCAP,&cap)==-1){
		printf("VIDIOC_QUERYCAP failed.\n");
	}
	printf("VIDIOC_QUERYCAP success.->DriverName:%s CardName:%s BusInfo:%s\n",\
		cap.driver,cap.card,cap.bus_info);//device info.设备信息    
		
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        printf("V4L2: %s is no video capture device\n", path);
        close(fd);
        return NULL;
    }
	
	fmtdesc.index = 0; //form number
	fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;//frame type  
	while(ioctl(fd,VIDIOC_ENUM_FMT,&fmtdesc) != -1){  
        //if(fmtdesc.pixelformat && fmt.fmt.pix.pixelformat){
            printf("VIDIOC_ENUM_FMT pixelformat:%s,%d\n",fmtdesc.description,fmtdesc.pixelformat);
        //}

		if(fmtdesc.pixelformat == CAM_PIX_FMT)
			index = fmtdesc.index;
		
        fmtdesc.index ++;
    }


	if(index == -1){
		printf("camero dont support YUYV format\n");
        close(fd);
        return NULL;
	}

    
	memset ( &fmt, 0, sizeof(fmt) );
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if (ioctl(fd,VIDIOC_G_FMT,&fmt) == -1) {
	   printf("VIDIOC_G_FMT failed.\n");
        close(fd);
        return NULL;
    }

  	printf("VIDIOC_G_FMT width %d, height %d, olorspace is %d\n",fmt.fmt.pix.width,fmt.fmt.pix.height,fmt.fmt.pix.colorspace);
	w = fmt.fmt.pix.width;
	h = fmt.fmt.pix.height;

	if(cam_width != w || cam_height != h){

		printf("wrong camera format, need be %dx%d\n",cam_width,cam_height);
		close(fd);
		return NULL;
	}
	
    //V4L2_PIX_FMT_RGB32   V4L2_PIX_FMT_YUYV   V4L2_STD_CAMERA_VGA  V4L2_PIX_FMT_JPEG
	fmt.fmt.pix.pixelformat = CAM_PIX_FMT;	
	if (ioctl(fd,VIDIOC_S_FMT,&fmt) == -1) {
	   printf("VIDIOC_S_FMT failed.\n");
        close(fd);
        return NULL;
    }


	if (ioctl(fd,VIDIOC_G_FMT,&fmt) == -1) {
	   printf("VIDIOC_G_FMT failed.\n");
        close(fd);
        return NULL;
    }
	 
	req.count = BUFF_COUNT;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;
	if ( ioctl(fd,VIDIOC_REQBUFS,&req)==-1){  
		printf("VIDIOC_REQBUFS map failed.\n");  
            close(fd);
            return NULL;
	} 


    memset(buffers,0,sizeof(*buffers)*req.count); 


	  
	for(i = 0; i < req.count; ++i)
	{  
		//struct v4l2_buffer buf;  
		memset(&buf[i],0,sizeof(buf[0])); 
		buf[i].index = i; 
		buf[i].type = V4L2_BUF_TYPE_VIDEO_CAPTURE;  
		buf[i].memory = V4L2_MEMORY_MMAP;  

		if(ioctl(fd,VIDIOC_QUERYBUF,&buf[i]) == -1)
		{ 
			printf("VIDIOC_QUERYBUF failed.\n");
            close(fd);
            return NULL;
		} 

  		//memory map
		buffers[i].length = buf[i].length;	
		buffers[i].start = mmap(NULL,buf[i].length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,buf[i].m.offset);  
		if(MAP_FAILED == buffers[i].start){  
			printf("memory map failed.\n");
            close(fd);
            return NULL;
		} 


		if (ioctl(fd , VIDIOC_QBUF, &buf[i]) ==-1) {
		    printf("VIDIOC_QBUF failed.->i=%d\n", i);
            close(fd);
            return NULL;
		}

	} 

	
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE; 
	if (ioctl(fd,VIDIOC_STREAMON,&type) == -1) {
		printf("VIDIOC_STREAMON failed.\n");
	    close(fd);
	    return NULL;
	}
 
    ioctl(fd, VIDIOC_QBUF, &buf[0]);
    
	cv::Mat yuvImg(cv::Size(w,h),CV_8UC2,buffers[0].start);
    yuvMat = yuvImg;


	v4l2_dev.fd = fd;
	
	pt = new unsigned char[VIDEO_WIDTH*VIDEO_HEIGHT*3];
	if(pt == NULL){
		printf("memory failed.\n");
	    close(fd);
	    return NULL;		
	}
	v4l2_dev.uvc_buf = pt;
	v4l2_dev.width = VIDEO_WIDTH;
	v4l2_dev.height = VIDEO_HEIGHT;
	v4l2_dev.uvc_en = uvc_en;
	v4l2_dev.map_buf = (unsigned char *)buffers[0].start;

	if(uvc_en)
		pthread_mutex_init(&v4l2_dev.mutex, NULL);
	
    return &v4l2_dev;
    
}  

int close_camera(void)
{
	if(v4l2_dev.uvc_buf)
		delete[] v4l2_dev.uvc_buf;

	if(v4l2_dev.fd)
		close(v4l2_dev.fd);

	return 0;
}

int get_capture_rgb(cv::Mat &rgbMat, struct V4L2_DEV *v4l2)
{

	if(v4l2->uvc_en){
		cv::Mat yuvImg(cv::Size(VIDEO_WIDTH,VIDEO_HEIGHT),CV_8UC2,v4l2_dev.uvc_buf);


		pthread_mutex_lock(&v4l2->mutex);

		cv::cvtColor(yuvImg, rgbMat, cv::COLOR_YUV2RGB_YVYU);
		pthread_mutex_unlock(&v4l2->mutex);
		
	}
	else{
	
	    ioctl(v4l2_dev.fd, VIDIOC_DQBUF, &buf[0]);
	    ioctl(v4l2_dev.fd, VIDIOC_QBUF, &buf[0]);
	    
	    
	    //cv::Mat yuvImg;
	    //cv::Mat yuvImg(cv::Size(VIDEO_WIDTH,VIDEO_HEIGHT),CV_8UC2,buffers[0].start);
	    //yuvImg.create(cy , cx, CV_8UC2);
	    //memcpy(yuvImg.data, data, len);
	    //cv::Mat rgbImg; 
	    
	    cv::cvtColor(yuvMat, rgbMat, cv::COLOR_YUV2RGB_YVYU);
	}
    return 0;
        
}



int get_capture_pic(struct V4L2_DEV *v4l2)
{
	static int cnt= 0;
	char buf[32];
    cv::Mat rgbMat;
	
	    ioctl(v4l2_dev.fd, VIDIOC_DQBUF, &buf[0]);
	    ioctl(v4l2_dev.fd, VIDIOC_QBUF, &buf[0]);
	    
	    
	    cv::cvtColor(yuvMat, rgbMat, cv::COLOR_YUV2BGR_YVYU);
		sprintf(buf,"cam_%d.jpg",cnt);
		cv::imwrite(buf,rgbMat);
	cnt++;

    return 0;
        
}

void *get_v4l2_buff(void)
{
	return buffers[0].start;
}
