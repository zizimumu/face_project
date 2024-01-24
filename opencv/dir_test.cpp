#include <stdio.h>
#include <stdlib.h>

#include<fcntl.h>

#include <stdlib.h>

#include <unistd.h>

#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;


static unsigned int dir_count = 0;
static unsigned int file_count = 0;
#define MAX_PATH 512  //最大文件长度定义为512

int fd_data,fd_lab,fd_class,weight,lenght;

void dirwalk(char *dir)
{
	char name[MAX_PATH];
	struct dirent *dp;
	DIR *dfd, *dfd_n;

	
	if((dfd = opendir(dir)) == NULL){
		fprintf(stderr, "dirwalk: can't open %s\n", dir);
		return;
	}
	
	while((dp = readdir(dfd)) != NULL){ //读目录记录项
		if(strcmp(dp->d_name, ".") == 0 || strcmp(dp -> d_name, "..") == 0){
			continue;  //跳过当前目录以及父目录
		}
		
		if(strlen(dir) + strlen(dp -> d_name) + 2 > sizeof(name)){
			fprintf(stderr, "dirwalk : name %s %s too long\n", dir, dp->d_name);
		}else{
			if( dp->d_type == DT_DIR){
				printf("dir : %s\n",dp->d_name);
				dir_count++;

				sprintf(name, "%s/%s", dir, dp->d_name);
				dirwalk(name);

			}
			if( dp->d_type == DT_REG){
				printf("file: %s, %d\n",dp->d_name,dir_count);

				sprintf(name, "%s/%s", dir, dp->d_name);
				Mat img = imread(name,IMREAD_GRAYSCALE);
				write(fd_data,img.data,img.cols*img.rows);
				write(fd_lab,&dir_count,sizeof(dir_count));

				weight = img.cols;
				lenght = img.rows;
			}

		}
	}
	
	closedir(dfd);
}



struct class_info{
	unsigned int start;
	unsigned int len;
	unsigned int lab;
};


int main(int argc, char *argv[])
{
	unsigned char *class_buf;
	unsigned int lab = 0,lab_pre = 0xffffffff;
	int ret,i,len,cnt,start;
    struct class_info class_s;


	fd_data = open("data.bin",O_RDWR |O_CREAT | O_TRUNC);
	fd_lab = open("lab.bin",O_RDWR|O_CREAT | O_TRUNC);
	fd_class = open("class.bin",O_RDWR|O_CREAT | O_TRUNC);


	if(argc == 1){
		dirwalk(".");//未加参数执行时，从当前目录开始遍历
	}else{
		while(--argc>0){
			dirwalk(*++argv);
		}
	}

    
    close(fd_data);
	close(fd_lab);

	//class_buf = malloc(sizeof(struct class_info)*dir_count);
	fd_lab = open("lab.bin",O_RDWR);

	len = 0;
	cnt = 0;
	start = 0;
	for(i=0;;i++){
		ret = read(fd_lab,&lab,4);
        
		if(ret < 4){
			if(len > 1){
				class_s.start = start;
				class_s.len = len+1;
				class_s.lab = lab_pre;
				write(fd_class,&class_s,sizeof(class_s));
			}
			break;
		}

		if(lab_pre == 0xffffffff){
		    lab_pre = lab;
		    start = i;
		    continue;
		}
        
		if(lab == lab_pre)
			len++;
		else{

			class_s.start = start;
			class_s.len = len+1;
			class_s.lab = lab_pre;
			write(fd_class,&class_s,sizeof(class_s));

			start = i;
			len = 0;
            
            
        	}
        	lab_pre = lab;
	}
	
    printf("class count %d, total lab %d,weight %d, lenght %d\n",dir_count,i,weight,lenght);
    close(fd_class);
    close(fd_lab);
    
	return 0;
}
