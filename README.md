# Glimpse

GPU-accelerated microarchitecture simulator for digital vision chips with integrated power and area modeling.

## Overview

Glimpse is a high-performance simulator designed for digital vision processing architectures. It provides  GPU-accelerated simulation capabilities for vision chip microarchitectures, with comprehensive power and area modeling using CACTI and McPAT.

## Features

- **High-performance simulation**: Via GPU-acceleration
- **Real-time Processing**: Live camera feed processing with OpenCV integration
- **VLIW Architecture Support**: Variable instruction-level parallelism (1-4 slots)
- **Pipelining Support**: 3-stage pipeline implementation
- **Power & Area Modeling**: Integrated CACTI-based power and area analysis
- **Flexible Bit Depth**: Support for 1-8 bit pixel processing

## Vision Algorithms Implemented

- Edge detection (1-bit)
- Prewitt edge detection (6-bit)
- Image smoothing (1-bit and 6-bit)
- Image thinning (1-bit)
- Binary belief propagation Ising model

## Requirements

### System Dependencies
- CUDA Toolkit (12.2.0 or later)
- OpenCV 4.x
- GCC 10.3.0 or later

### Hardware Requirements
- NVIDIA GPU with compute capability 7.5+
- Camera (for real-time processing)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd glimpse
   ```

2. **Install dependencies**:
   - Install CUDA Toolkit
   - Install OpenCV 4.x development libraries
   - Ensure pkg-config can find OpenCV
   - If on Imperial College London lab machines, run `source setup_project.sh` instead

3. **Build the project**:
   ```bash
   make clean all
   ```

## Usage

### Basic Usage

Run all test programs on an image:
```bash
./build/main
```

### Custom Configuration

```bash
./build/main [OPTIONS]
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --image` | Input image file | `images/whitecat_600.jpg` |
| `-d, --dimension` | Image dimension (square) | `128` |
| `-p, --program` | Program file (.vis) | `programs/1_vliw_slot/edge_detection_one_bit.vis` |
| `-g, --use-gpu` | Enable GPU acceleration | `false` |
| `-w, --vliw-width` | VLIW width (1-4) | `1` |
| `-r, --real-time` | Real-time camera processing | `false` |
| `-l, --pipelining` | Enable pipelining | `false` |
| `-b, --bits` | Bits per pixel (1-8) | `1` |
| `-t, --twos-complement` | Two's complement output | `false` |
| `--display-dimension` | Display window size | `1000` |
| `-h, --help` | Show help message | - |

### Example Commands

**GPU-accelerated real-time edge detection**:
```bash
./build/main --dimension 1024 --use-gpu --real-time --program programs/1_vliw_slot/edge_detection_one_bit.vis --bits 1
```

**High-resolution belief propagation**:
```bash
./build/main --dimension 512 --use-gpu --real-time --program programs/1_vliw_slot/binary_bp_ising_model.vis --bits 8
```

**Batch processing with custom image**:
```bash
./build/main --image images/custom.jpg --dimension 256 --use-gpu
```

## Program Files

Vision programs are located in the `programs/` directory, organized by VLIW width:

- `1_vliw_slot/` - Single instruction slot programs
- `2_vliw_slot/` - Dual instruction slot programs  
- `3_vliw_slot/` - Triple instruction slot programs
- `4_vliw_slot/` - Quad instruction slot programs
- `pipelining/` - Pipelined implementations

### Available Programs

- `edge_detection_one_bit.vis` - Simple edge detection
- `prewitt_edge_detection_*.vis` - Prewitt edge detection
- `smoothing_*.vis` - Image smoothing filters
- `thinning_one_bit.vis` - Morphological thinning
- `binary_bp_ising_model.vis` - Belief propagation

## Output

The simulator generates:

1. **Performance Metrics**:
   - Processing time per frame
   - Frame rate (FPS)
   - Average chip performance (μs)
   - Instruction count and utilization

2. **Power Analysis**:
   - Dynamic power consumption
   - Leakage power (subthreshold and gate)
   - Total power (W)

3. **Area Analysis**:
   - Compute area (μm²)
   - Memory area (μm²)
   - Total chip area

4. **Processed Images**:
   - Output images saved to `outputimages/` directory
   - Original quantized images for comparison

## Architecture Details

### Processing Elements
- Configurable VLIW width (1-4 instruction slots)
- Local memory per processing element
- Carry register support
- Neighbor communication capabilities

### Memory Hierarchy
- Local memory per PE (configurable size)
- Shared neighbor values
- External output storage

### Instruction Set
- Binary operations with carry support
- Memory addressing modes
- Neighbor data access (up, down, left, right)
- Photodiode (PD) input access

## Development

### Project Structure
```
glimpse/
├── src/                    # Source code
│   ├── main.cu            # Main application
│   ├── isa.cu/h           # Instruction set architecture
│   ├── pe.cu/h            # Processing element implementation
│   ├── powerandarea.cu/h  # Power and area modeling
│   └── utils/             # Utility functions
├── programs/              # Vision algorithm programs
├── images/                # Test images
├── cacti/                 # CACTI power/area modeling tool
├── notes/                 # Documentation and BNF grammars
└── outputimages/          # Generated output images
```

### Building for Development

For debugging builds, modify the Makefile to use:
```makefile
CFLAGS = -arch=sm_75 -Xptxas -O1 -Xcompiler -O1 -use_fast_math -rdc=true
```

## Troubleshooting

**CUDA Errors**: Ensure CUDA toolkit is properly installed and GPU compute capability is supported.

**OpenCV Issues**: Verify OpenCV 4.x is installed with development headers and pkg-config can locate it.

**Camera Access**: For real-time mode, ensure camera permissions and V4L2 support on Linux.

**Memory Issues**: Large images or high VLIW widths may require significant GPU memory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions including bug fixes, performance improvements, and documentation updates. Please submit a pull request to contibute!