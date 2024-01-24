#ifndef __V4L2_DEV_H_
#define __V4L2_DEV_H_



struct V4L2_DEV{
	int fd;
	unsigned char *uvc_buf;
	unsigned char *map_buf;
	int uvc_en;
	unsigned int width;
	unsigned int height;
	pthread_mutex_t mutex;

};






#endif
