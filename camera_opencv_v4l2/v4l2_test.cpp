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


struct buffer{  
	void *start;  
	unsigned int length;  
}; 

#define BUFF_COUNT 1
#define CAM_PIX_FMT V4L2_PIX_FMT_YUYV
#define  VIDEO_WIDTH  640
#define  VIDEO_HEIGHT  480
#define DEV_PATH "/dev/video0" //0/9 
#define CAPTURE_FILE "yuv_raw.bin"

using namespace cv;


int main()
{  
#if 1
	//1.open device.打开摄像头设备 
	int index = -1;
    struct buffer buffers[BUFF_COUNT];
    struct v4l2_buffer buf[BUFF_COUNT]; 
    
    
	int fd = open(DEV_PATH,O_RDWR,0);//弄了好久 以阻塞模式打开摄像头  | O_NONBLOCK 非堵塞
	if(fd<0){
		printf("open device failed.\n");
	}
 

    //2.search device property.查询设备属性

	struct v4l2_capability cap;
	if(ioctl(fd,VIDIOC_QUERYCAP,&cap)==-1){
		printf("VIDIOC_QUERYCAP failed.\n");
	}
	printf("VIDIOC_QUERYCAP success.->DriverName:%s CardName:%s BusInfo:%s\n",\
		cap.driver,cap.card,cap.bus_info);//device info.设备信息    
		

	//3.show all supported format.显示所有支持的格式
	struct v4l2_fmtdesc fmtdesc;
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
		return 0;
	}

    struct v4l2_format fmt;
	memset ( &fmt, 0, sizeof(fmt) );
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if (ioctl(fd,VIDIOC_G_FMT,&fmt) == -1) {
	   printf("VIDIOC_G_FMT failed.\n");
	   return -1;
    }

  	printf("VIDIOC_G_FMT width %ld, height %d, olorspace is %ld\n",fmt.fmt.pix.width,fmt.fmt.pix.height,fmt.fmt.pix.colorspace);
	
	
    //V4L2_PIX_FMT_RGB32   V4L2_PIX_FMT_YUYV   V4L2_STD_CAMERA_VGA  V4L2_PIX_FMT_JPEG
	fmt.fmt.pix.pixelformat = CAM_PIX_FMT;	
	if (ioctl(fd,VIDIOC_S_FMT,&fmt) == -1) {
	   printf("VIDIOC_S_FMT failed.\n");
	   return -1;
    }


	if (ioctl(fd,VIDIOC_G_FMT,&fmt) == -1) {
	   printf("VIDIOC_G_FMT failed.\n");
	   return -1;
    }
  	//printf("VIDIOC_G_FMT sucess.->fmt.fmt.width is %ld\nfmt.fmt.pix.height is %ld\n\
	//	fmt.fmt.pix.colorspace is %ld\n",fmt.fmt.pix.width,fmt.fmt.pix.height,fmt.fmt.pix.colorspace);


	//6.1 request buffers.申请缓冲区
	struct v4l2_requestbuffers req;  
	req.count = BUFF_COUNT;//frame count.帧的个数 
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;//automation or user define．自动分配还是自定义
	if ( ioctl(fd,VIDIOC_REQBUFS,&req)==-1){  
		printf("VIDIOC_REQBUFS map failed.\n");  
		close(fd);  
		exit(-1);  
	} 

	//6.2 manage buffers.管理缓存区
	//应用程序和设备３种交换方式：read/write，mmap，用户指针．这里用memory map.内存映射
	unsigned int i = 0;  
	//buffers = (struct buffer*) calloc (req.count, sizeof(*buffers)); 
	//buffers = (struct buffer *)calloc(req.count, sizeof(*buffers));

    memset(buffers,0,sizeof(*buffers)*req.count); 


	  
	for(i = 0; i < req.count; ++i)
	{  
		//struct v4l2_buffer buf;  
		memset(&buf[i],0,sizeof(buf[0])); 
		buf[i].index = i; 
		buf[i].type = V4L2_BUF_TYPE_VIDEO_CAPTURE;  
		buf[i].memory = V4L2_MEMORY_MMAP;  
		//查询序号为n_buffers 的缓冲区，得到其起始物理地址和大小
		if(ioctl(fd,VIDIOC_QUERYBUF,&buf[i]) == -1)
		{ 
			printf("VIDIOC_QUERYBUF failed.\n");
			close(fd);  
			exit(-1);  
		} 

  		//memory map
		buffers[i].length = buf[i].length;	
		buffers[i].start = mmap(NULL,buf[i].length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,buf[i].m.offset);  
		if(MAP_FAILED == buffers[i].start){  
			printf("memory map failed.\n");
			close(fd);  
			exit(-1);  
		} 

		//Queen buffer.将缓冲帧放入队列 
		if (ioctl(fd , VIDIOC_QBUF, &buf[i]) ==-1) {
		    printf("VIDIOC_QBUF failed.->i=%d\n", i);
		    return -1;
		}

	} 

	//7.使能视频设备输出视频流
	enum v4l2_buf_type type; 
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE; 
	if (ioctl(fd,VIDIOC_STREAMON,&type) == -1) {
		printf("VIDIOC_STREAMON failed.\n");
		return -1;
	}
    
    
//	//8.DQBUF.取出一帧
//    struct v4l2_buffer buf1;  
//	memset(&buf1, 0, sizeof(buf1)); 
//    buf1.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;  
//    buf1.memory = V4L2_MEMORY_MMAP;
//    if (ioctl(fd, VIDIOC_DQBUF, &buf1) == -1) {
//        printf("VIDIOC_DQBUF failed.->fd=%d\n",fd);
//        return -1;
//    }  

    //cv::Mat yuvImg;
    cv::Mat yuvImg(cv::Size(VIDEO_WIDTH,VIDEO_HEIGHT),CV_8UC2,buffers[0].start);
    //yuvImg.create(cy , cx, CV_8UC2);
    //memcpy(yuvImg.data, data, len);
    cv::Mat rgbImg; 
    
	ioctl(fd, VIDIOC_QBUF, &buf[0]);
	ioctl(fd, VIDIOC_DQBUF, &buf[0]); // VIDIOC_DQBUF will wait for video ready


    cv::namedWindow("uvc");
    while (1)
    {
        
        

        
        ioctl(fd, VIDIOC_QBUF, &buf[0]);
        cv::cvtColor(yuvImg, rgbImg, cv::COLOR_YUV2RGB_YVYU);
        
        
        cv::imshow("uvc",rgbImg);
        //Esc
        if (cv::waitKey(1) == 27)
        {
            break;
        }
        ioctl(fd, VIDIOC_DQBUF, &buf[0]);
    }	
	
	while(1);
#else

   namedWindow("Win7x64",WINDOW_NORMAL);
   VideoCapture capture;
   Mat camera;
   //采用 Directshow 的方式打开第一个摄像头设备。
   capture.open(0,CAP_V4L2);
   if(!capture.isOpened())
   {
       return -1;
   }
   capture.read(camera);
   
   printf("get pic, w %d, h %d\n",camera.cols,camera.rows);
   while (true)
   {
       //读取一帧图像
       capture.read(camera);
       if(camera.empty())
       {
           continue;
       }
       imshow("Win7x64",camera);
       //Esc
       if (waitKey(1) == 27)
       {
           break;
       }
   }
	return 0;
#endif
}  
