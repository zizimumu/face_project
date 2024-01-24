CCXX=arm-linux-gnueabihf-g++

CFLAGS_=-I/home/emy/armnn-tflite/armnn-dist/armnn/include
CFLAGS_ +=-I/home/emy/armnn-tflite/armnn-dist/boost/include
CFLAGS_ +=-I/home/emy/armnn-tflite/armnn/src/armnnUtils


LDFLAGS_= -L/home/emy/armnn-tflite/armnn-dist/armnn/lib
LDFLAGS_+= -L/home/emy/armnn-tflite/armnn-dist/boost/lib
#the libs must be put after the cpp files
#LIBS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -lopencv_objdetect -lopencv_xfeatures2d -lopencv_objdetect -lopencv_xobjdetect
LIBS=-larmnn -larmnnTfLiteParser -lprotobuf -lpthread

all:
	$(CCXX)  -std=c++14  $(CFLAGS_)  -Wall    -Wextra read_mobilefacenet_non_quant.cpp  -o read_mobilefacenet_non_quant $(LDFLAGS_) $(LIBS) 


.PHONY: clean
clean:
	rm -f *.o
