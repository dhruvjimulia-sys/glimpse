NVCC = nvcc

CFLAGS = -rdc=true -Xcudafe "--diag_suppress=170 --diag_suppress=550 --diag_suppress=1675"

SRC = $(wildcard src/*.cu) $(wildcard src/**/*.cu) $(wildcard src/*.cpp) $(wildcard src/**/*.cpp)

TARGET = build/main

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC) 

clean:
	rm -f $(TARGET)

.PHONY: all clean