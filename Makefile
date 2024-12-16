NVCC = nvcc

CFLAGS = 

SRC = $(wildcard src/*.cu) 

TARGET = build/main

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean