NVCC = nvcc

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS  = $(shell pkg-config --libs opencv4)

# suppressed warnings from SBT library --diag_suppress=170 --diag_suppress=550 --diag_suppress=1675
# suppressed warnings from OpenCV --diag_suppress=611

# optimized build
# CFLAGS = -arch=sm_61 -Xptxas -O3 -Xcompiler -O3 -use_fast_math -rdc=true -Xcudafe "--diag_suppress=170 --diag_suppress=550 --diag_suppress=1675"

# debugging build
# Computing Labs: CFLAGS = -arch=sm_75 -Xptxas -O1 -Xcompiler -O1 -use_fast_math -rdc=true -Xcudafe "--diag_suppress=170 --diag_suppress=550 --diag_suppress=1675"
CFLAGS = -arch=sm_75 -Xptxas -O1 -Xcompiler -O1 -use_fast_math -rdc=true -Xcudafe "--diag_suppress=170 --diag_suppress=550 --diag_suppress=1675 --diag_suppress=611"

# debugging + address sanitizer
# CFLAGS = -arch=sm_61 -Xcompiler -fsanitize=address -Xcompiler -fsanitize=undefined -g -O1 -use_fast_math -rdc=true -Xcudafe "--diag_suppress=170 --diag_suppress=550 --diag_suppress=1675"

SRC = $(wildcard src/*.cu) $(wildcard src/**/*.cu) $(wildcard src/*.cpp) $(wildcard src/**/*.cpp)

TARGET = build/main

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC) $(OPENCV_CFLAGS) $(OPENCV_LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean