#include <stdio.h>      /*标准输入输出定义*/
#include <stdlib.h>
#include <unistd.h>     /*Unix标准函数定义*/
#include <sys/types.h>  /**/
#include <sys/stat.h>   /**/
#include <fcntl.h>      /*文件控制定义*/
#include <termios.h>    /*PPSIX终端控制定义*/
#include <errno.h>      /*错误号定义*/
#include <getopt.h>
#include <string.h>

#define FALSE 1
#define TRUE 0



void print_usage();

int speed_arr[] = { 
	B921600, B460800, B230400, B115200, B57600, B38400, B19200, 
	B9600, B4800, B2400, B1200, B300, 
};

int name_arr[] = {
	921600, 460800, 230400, 115200, 57600, 38400,  19200,  
	9600,  4800,  2400,  1200,  300,  
};

void set_speed(int fd, int speed)
{
	int   i;
	int   status;
	struct termios   Opt;
	tcgetattr(fd, &Opt);

	for ( i= 0;  i < sizeof(speed_arr) / sizeof(int);  i++) {
		if  (speed == name_arr[i])	{
			tcflush(fd, TCIOFLUSH);
			cfsetispeed(&Opt, speed_arr[i]);
			cfsetospeed(&Opt, speed_arr[i]);
			status = tcsetattr(fd, TCSANOW, &Opt);
			if  (status != 0)
				perror("tcsetattr fd1");
				return;
		}
		tcflush(fd,TCIOFLUSH);
  	 }

	//if (i == 12){
	//	printf("\tSorry, please set the correct baud rate!\n\n");
	//	print_usage(stderr, 1);
	//}
}

int set_Parity(int fd,int databits,int stopbits,int parity,int stall_len)
{
	struct termios options;
	if  ( tcgetattr( fd,&options)  !=  0) {
		perror("SetupSerial 1");
		return(FALSE);
	}
	options.c_cflag &= ~CSIZE ;
	switch (databits) /*设置数据位数*/ {
	case 7:
		options.c_cflag |= CS7;
	break;
	case 8:
		options.c_cflag |= CS8;
	break;
	default:
		fprintf(stderr,"Unsupported data size\n");
		return (FALSE);
	}
	
	switch (parity) {
	case 'n':
	case 'N':
		options.c_cflag &= ~PARENB;   /* Clear parity enable */
		options.c_iflag &= ~INPCK;     /* Enable parity checking */
	break;
	case 'o':
	case 'O':
		options.c_cflag |= (PARODD | PARENB);  /* 设置为奇效验*/
		options.c_iflag |= INPCK;             /* Disnable parity checking */
	break;
	case 'e':
	case 'E':
		options.c_cflag |= PARENB;     /* Enable parity */
		options.c_cflag &= ~PARODD;   /* 转换为偶效验*/ 
		options.c_iflag |= INPCK;       /* Disnable parity checking */
	break;
	case 'S':	
	case 's':  /*as no parity*/
		options.c_cflag &= ~PARENB;
		options.c_cflag &= ~CSTOPB;
	break;
	default:
		fprintf(stderr,"Unsupported parity\n");
		return (FALSE);
	}
 	/* 设置停止位*/  
  	switch (stopbits) {
   	case 1:
    	options.c_cflag &= ~CSTOPB;
  	break;
 	case 2:
  		options.c_cflag |= CSTOPB;
  	break;
 	default:
  		fprintf(stderr,"Unsupported stop bits\n");
  		return (FALSE);
 	}
  	/* Set input parity option */
  	if (parity != 'n')
    	options.c_iflag |= INPCK;


	options.c_oflag &= ~(BSDLY|CRDLY|FFDLY|NLDLY|OFDEL|OFILL|OLCUC|ONLRET|ONOCR|OPOST|OCRNL|ONLCR);
	options.c_lflag &= ~(FLUSHO|ECHOKE|PENDIN|TOSTOP|XCASE|ECHO|ECHOK|ECHONL|ISIG|IEXTEN|ECHOE);
	options.c_lflag |= ICANON;
	options.c_iflag &= ~(IXON|IXOFF|IXANY|IGNCR|ICRNL|INLCR|BRKINT|IGNPAR|IMAXBEL|IUCLC|PARMRK|IGNBRK|INPCK|ISTRIP);
	options.c_cc[VKILL]= _POSIX_VDISABLE;
	options.c_cc[VERASE] = _POSIX_VDISABLE;
	options.c_cc[VEOL] = _POSIX_VDISABLE;
	options.c_cc[VEOL2] = _POSIX_VDISABLE;
	options.c_cc[VEOF] = _POSIX_VDISABLE;
	options.c_cc[VWERASE] = _POSIX_VDISABLE;
	options.c_cc[VREPRINT] = _POSIX_VDISABLE;
	

  	tcflush(fd,TCIFLUSH); /* Update the options and do it NOW */
  	if (tcsetattr(fd,TCSANOW,&options) != 0) {
    	perror("SetupSerial 3");
  		return (FALSE);
 	}
	return (TRUE);
}

/**
	*@breif 打开串口
*/
int OpenDev(char *Dev)
{
	int fd = open( Dev, O_RDWR );         //| O_NOCTTY | O_NDELAY
 	if (-1 == fd) { /*设置数据位数*/
   		perror("Can't Open Serial Port");
   		return -1;
	} else
		return fd;
}


/* The name of this program */
const char * program_name;

/* Prints usage information for this program to STREAM (typically
 * stdout or stderr), and exit the program with EXIT_CODE. Does not
 * return.
 */


void print_hex(unsigned char *buf,int len)
{
	int i;
	for(i=0;i<len;i++){
		if((i%16) == 0)
			printf("\n");
		printf("  0x%02x",buf[i]);
	}
    printf("\n");
	
}


#if 1
 char buff[] = "1234567890abcdefghijklmnopqrstuvwxyz.!@#$%^&*()_+\n";
char rev_buf[512]; 
int main(int argc, char *argv[])
{
	int  fd, next_option, havearg = 0;
	char *device;
	int i=0,j=0,ret,valid,remain;
	int nread,count= 0,total;			/* Read the counts of data */
		/* Recvice data buffer */
	pid_t pid;
	unsigned int correct_cnt = 0;
	
    char *read_p;
	int speed;
	
	
	const char *const short_options = "hd:l:t:s:b:";

	const struct option long_options[] = {
		{ "help",   0, NULL, 'h'},
		{ "device", 1, NULL, 'd'},
        { "len", 1, NULL, 'l'},
        { "timer", 1, NULL, 't'},
		{ "string", 1, NULL, 's'},
		{ "baudrate", 1, NULL, 'b'},

		{ NULL,     0, NULL, 0  }
	};
	
	program_name = argv[0];

	do {
		next_option = getopt_long (argc, argv, short_options, long_options, NULL);
		switch (next_option) {
			case 'h':
				//print_usage (stdout, 0);
			case 'd':
				device = optarg;
				havearg = 1;
				break;
			case 'b':
				speed = atoi(optarg);
				break;

			case -1:
				if (havearg)  break;
			case '?':
				//print_usage (stderr, 1);
			default:
				abort ();
		}
	}while(next_option != -1);


    
	fd = OpenDev(device);

	if (fd > 0) {
		set_speed(fd, speed);
	} else {
		fprintf(stderr, "Error opening %s: %s\n", device, strerror(errno));
		exit(1);
	}

	if (set_Parity(fd,8,1,'N',0)== FALSE) {
		fprintf(stderr, "Set Parity Error\n");
		close(fd);
		exit(1);
	}


#if 1
	
	printf("\nstart: write %d\n",strlen(buff));
	write(fd,buff,strlen(buff));

		while(1) {
			nread = read(fd, rev_buf, sizeof(rev_buf));
			if(nread > 0)
				printf("%s",rev_buf);
           
		}	


#endif

#if 0
	pid = fork();	
	

	if (pid < 0) { 
		fprintf(stderr, "Error in fork!\n"); 
	} else if (pid == 0){

		
		while(1) {
			sleep(2);
            write(fd,buff,strlen(buff));
		}
		exit(0);
	} else { 

		while(1) {
			nread = read(fd, rev_buf, sizeof(rev_buf));
			if(nread > 0)
				printf("%s",rev_buf);
           
		}	
	}
#endif
	close(fd);
	exit(0);
}

#endif 

int uart_init(char *dev)
{
    int fd;
 	fd = OpenDev(dev);

	if (fd > 0) {
		set_speed(fd, 115200);
	} else {
		
        close(fd);
		return -1;
	}

	if (set_Parity(fd,8,1,'N',0)== FALSE) {

		close(fd);
		return -1;
	}  
    
    return fd;
    
}

