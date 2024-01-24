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

	if (i == 12){
		printf("\tSorry, please set the correct baud rate!\n\n");
		print_usage(stderr, 1);
	}
}
/*
	*@brief   设置串口数据位，停止位和效验位
	*@param  fd     类型  int  打开的串口文件句柄*
	*@param  databits 类型  int 数据位   取值 为 7 或者8*
	*@param  stopbits 类型  int 停止位   取值为 1 或者2*
	*@param  parity  类型  int  效验类型 取值为N,E,O,,S
*/
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

void print_usage (FILE *stream, int exit_code)
{
    fprintf(stream, "Usage: %s option [ dev... ] \n", program_name);
    fprintf(stream,
            "\t-h  --help     Display this usage information.\n"
            "\t-d  --device   The device ttyS[0-3] or ttySCMA[0-1]\n"
	    "\t-b  --baudrate Set the baud rate you can select\n" 
	    "\t               [230400, 115200, 57600, 38400, 19200, 9600, 4800, 2400, 1200, 300]\n"
            "\t-s  --string   Write the device data\n");
    exit(exit_code);
}

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


/*
	*@breif  main()
 */
 char buff[1024];	
 char xmit[] = "write thread start\n"; /* Default send data */ 
#define START_FLAT "AT+"
#define MAX_CMD_LEN 32
int string_compare(char *src, char *src2, int n)
{
	int i;
	for(i=0;i<n;i++){
		if(src[i] != src2[])
			return -1;
	}

	return 0;
}


#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

typedef int cmd_func(char *, int);

struct _cmd_list_ {
	cmd_func func;
	char cmd[MAX_CMD_LEN];
};



unsigned char password[6];
int passwd_func(char *cmd, int len)
{
	if((len != 20) || (cmd[6] != '=' ) || cmd[13]) {
		printf("wrong format for setting pass word,e.g. passwd=654321+123456\n");
		return -1;
	}
}

struct _cmd_list_ cmd_list[] = {

	{passwd_func,"passwd"},
};

comd_parse(char *buf, int len)
{
	char end;
	char cmd_buf[MAX_CMD_LEN];
	int cmd_len = 0;
	int ret = 0;

	if(len < 5 || len > MAX_CMD_LEN|| string_compare(START_FLAT,buf,strlen(START_FLAT))){
		goto unknow;
	}

	end = buf[len-1];
	if((end != 0x0d) && (end != 0x0a){
		goto unknow;
	}

	//delete "\r\n"
	buf[len-1] = 0;
	cmd_len = len - strlen(START_FLAT) - 1;
	
	if((buf[len-2] == 0x0d) || (buf[len-2] != 0x0a)
		cmd_len--;

	
	if(cmd_len > 0){
		memcpy(cmd_buf,buf+strlen(START_FLAT),cmd_len);

		n = ARRAY_SIZE(cmd_list);
		for(i = 0; i< n;i++){
			string_compare(cmd_buf,(char *)cmd_list[i],cmd_len)
		}
	}
	
unknow:
	buf[len-1] = 0;
	printf("unknow command %s\n",buf);
	return -1;
	
		
}
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
				print_usage (stdout, 0);
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
				print_usage (stderr, 1);
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

	pid = fork();	
	

	if (pid < 0) { 
		fprintf(stderr, "Error in fork!\n"); 
	} else if (pid == 0){

		write(fd,xmit,sizeof(xmit));
		while(1) {
			sleep(1);
		}
		exit(0);
	} else { 

		while(1) {
			nread = read(fd, buff, sizeof(buff));
			if(nread > 0)
				print_hex(buff,nread);
           
		}	
	}
	close(fd);
	exit(0);
}

