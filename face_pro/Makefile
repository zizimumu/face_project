CCXX=arm-linux-gnueabihf-g++
CC=arm-linux-gnueabihf-gcc

CFLAGS_=-I/home/emy/source/armnn/armnn-dist/armnn/include
CFLAGS_ +=-I/home/emy/source/armnn/armnn-dist/boost/include
CFLAGS_ +=-I/home/emy/source/armnn/armnn-dist/armnn/include/armnnUtils

CFLAGS_+=-I /home/emy/source/SeetaFaceEngine-master/FaceDetection/include
CFLAGS_+=-I /home/emy/source/SeetaFaceEngine-master/FaceAlignment/include

CFLAGS_+=-I /home/emy/source/opencv-4.1.0/__install_armv7hf_fastmath/include/opencv4


LDFLAGS_= -L/home/emy/source/armnn/armnn-dist/armnn/lib
LDFLAGS_+= -L/home/emy/source/armnn/armnn-dist/boost/lib

LDFLAGS_+= -L/home/emy/source/SeetaFaceEngine-master/FaceDetection/__build_armv7hf
LDFLAGS_+= -L/home/emy/source/SeetaFaceEngine-master/FaceAlignment/__build_armv7hf

LDFLAGS_+=-L/home/emy/source/opencv-4.1.0/__install_armv7hf_fastmath/lib

#the libs must be put after the cpp files
LIBS= -lseeta_facedet_lib -lseeta_fa_lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_flann -lopencv_imgproc -lopencv_features2d   -lopencv_calib3d
LIBS+=-larmnn -larmnnTfLiteParser -lprotobuf -lpthread


all:
	${CC} -c md5/md5.c 
	${CC} -c uvc-gadget.c 
	${CC} -c uart.c 
	$(CCXX)  -std=c++14 -O3 $(CFLAGS_)  -Wall  -Wno-unused-variable  -Wextra face.cpp  v4l2_test.cpp read_config.cpp md5.o uvc-gadget.o uart.o -o face $(LDFLAGS_) $(LIBS) 
#	$(CCXX)  -std=c++14 -O3 $(CFLAGS_)  -Wall  -Wno-unused-variable  -Wextra face_time.cpp  v4l2_test.cpp md5.o uvc-gadget.o -o face_time $(LDFLAGS_) $(LIBS) 


.PHONY: clean
clean:
	rm -f *.o
