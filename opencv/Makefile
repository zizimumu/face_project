CCXX=g++
CFLAGS_=-I /home/emy/source/opencv-4.1.0/__install/include/opencv4/
#CFLAGS_ += -I /opt/poky-atmel/2.4.2/sysroots/cortexa5hf-vfp-poky-linux-gnueabi/usr/include/libdrm/
LDFLAGS_=-L /home/emy/source/opencv-4.1.0/__install/lib

#the libs must be put after the cpp files
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -lopencv_objdetect -lopencv_xfeatures2d -lopencv_objdetect -lopencv_xobjdetect

all:
	$(CCXX)  -std=c++11 $(LDFLAGS_) $(CFLAGS_)   test.cpp $(LIBS) -o test


.PHONY: clean
clean:
	rm -f *.o
