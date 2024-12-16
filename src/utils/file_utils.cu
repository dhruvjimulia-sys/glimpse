#include "file_utils.h"

void readFile(std::string & filename, std::string & fileContents) {
    std::ifstream file(filename);
    if (file) {
        std::stringstream buffer;
        buffer << file.rdbuf();      // Read the file into the buffer
        fileContents = buffer.str(); // Convert buffer to string
    } else {
        std::cerr << "Error opening file: " << filename << std::endl;
    }
}