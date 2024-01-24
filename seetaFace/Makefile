CCXX=g++
CFLAGS_=-I /home/emy/source/SeetaFaceEngine-master/FaceDetection/include
CFLAGS_+=-I /home/emy/source/SeetaFaceEngine-master/FaceAlignment/include

#CFLAGS_ += -I /opt/poky-atmel/2.4.2/sysroots/cortexa5hf-vfp-poky-linux-gnueabi/usr/include/libdrm/
#LDFLAGS_=-L  /home/emy/source/SeetaFaceEngine-master/FaceDetection/__build_PC
LDFLAGS_=-L /home/emy/source/SeetaFaceEngine-master/FaceDetection/__build_PC
#the libs must be put after the cpp files
LIBS= -lseeta_facedet_lib -lseeta_fa_lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -lopencv_objdetect -lopencv_xfeatures2d -lopencv_objdetect -lopencv_xobjdetect -lopencv_calib3d

all:
	$(CCXX)  -std=c++11 $(LDFLAGS_) $(CFLAGS_)   face_alignment_test.cpp $(LIBS) -o face_alignment_test


.PHONY: clean
clean:
	rm -f *.o
