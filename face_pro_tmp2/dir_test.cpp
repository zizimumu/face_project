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

/*
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;

*/
using namespace std;



#define MAX_PATH 512  //最大文件长度定义为512



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
		

		if( dp->d_type == DT_DIR){
			printf("dir skip: %s\n",dp->d_name);
			continue;

		}
		if( dp->d_type == DT_REG){
			if(strlen(dp -> d_name) > MAX_PATH){
				printf("file name %s too long\n",dp->d_name);
				continue;
			}
			else{

			}

			//sprintf(name, "%s/%s", dir, dp->d_name);
			//Mat img = imread(name,IMREAD_GRAYSCALE);

		}


	}
	
	closedir(dfd);
}




int main(int argc, char *argv[])
{


	if(argc == 1){
		dirwalk(".");//未加参数执行时，从当前目录开始遍历
	}else{

		dirwalk(*++argv);

	}

    
	return 0;
}
