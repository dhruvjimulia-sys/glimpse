NVCC = nvcc

CFLAGS = 

SRC = $(wildcard src/*.cu)  $(wildcard src/*.cpp)

TARGET = build/main

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean