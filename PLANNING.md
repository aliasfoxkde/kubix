# KUBIX: 3D Nonogram-Inspired Compression Algorithm

## Project Overview

KUBIX is an experimental compression algorithm that transforms binary data into 3D cubes and uses axis sum projections for compression, inspired by 3D nonogram puzzles. The algorithm splits binary files into fixed-size 3D cubes, calculates sum totals along X, Y, and Z axes, and stores only these projections as compressed data. Reconstruction involves solving for valid cube configurations that match the stored axis sums.

**Copyright Notice**: KUBIX (TM) Copyright (C) 2015. Micheal L. C. Kinney. All Rights Reserved.

## Theoretical Foundation

### Core Algorithm Concept

1. **Data Transformation**: Binary data is segmented into 3D cubes of size n³ bits
2. **Axis Projection**: For each cube, calculate sum totals for:
   - X-axis projections (row sums across YZ planes)
   - Y-axis projections (column sums across XZ planes)
   - Z-axis projections (layer sums across XY planes)
3. **Compression**: Store only the axis sum arrays instead of the full cube data
4. **Reconstruction**: Solve the constraint satisfaction problem to find valid cube configurations

### Mathematical Basis

The algorithm leverages the mathematical relationship between 3D binary data and its orthogonal projections:
- Given a 3D binary cube C[x,y,z] where each element is 0 or 1
- X-sums: Σ(C[i,j,k]) for all j,k where i is fixed
- Y-sums: Σ(C[i,j,k]) for all i,k where j is fixed
- Z-sums: Σ(C[i,j,k]) for all i,j where k is fixed

The compression ratio depends on the relationship between the original cube size (n³ bits) and the projection storage (3×n integers).

## Technical Implementation

### Current Architecture

#### File Structure
```
kubix-v1/
├── kubix-alt.py              # Main modern implementation (64³ cubes)
├── kubix-compressor-v0.33.min.py  # Legacy compressor (4³ cubes)
├── kubix-decomp-v82.py       # Legacy decompressor with validation
├── genHTML.py                # Visualization generator
├── libs/
│   ├── kbx.py               # Core algorithm utilities
│   ├── timing.py            # Performance measurement
│   └── ficle.py             # File handling utilities
├── Test/
│   └── unit_tests.py        # Validation framework
├── HTML_Outputs/            # Generated puzzle visualizations
├── Output/                  # Compressed file outputs (.kbx, .kby, .kbz)
└── Samples/                 # Test video files
```

#### Core Components

**1. Data Chunking (`file_to_cubes`)**
- Reads binary files and splits into fixed-size chunks
- Pads incomplete chunks with null bytes
- Reshapes byte arrays into 3D numpy arrays
- Current implementation uses 64³ cubes (262,144 bytes per cube)

**2. Axis Sum Calculation (`cube_to_sums`)**
```python
x_sums = np.sum(cube, axis=(1, 2))  # Sum across Y,Z for each X
y_sums = np.sum(cube, axis=(0, 2))  # Sum across X,Z for each Y
z_sums = np.sum(cube, axis=(0, 1))  # Sum across X,Y for each Z
```

**3. Compression Pipeline**
- Process each cube independently
- Store axis sums as separate .npy files
- Compress using ZIP with LZMA algorithm
- Generate metadata (cube size, number of cubes)

**4. Reconstruction Algorithm (`reconstruct_valid_cube`)**
- **Greedy Initialization**: Set bits where all three axis constraints allow
- **Iterative Refinement**: Flip bits to minimize axis sum differences
- **Constraint Satisfaction**: Limited iterations (20) to find valid solution
- **Early Termination**: Exit when all axis sums match exactly

### Key Technical Challenges

#### 1. Multiple Valid Solutions Problem
Like nonogram puzzles, multiple cube configurations can satisfy the same axis sums. The current algorithm finds *a* valid solution but may not recover the *original* solution.

**Current Approach**: Greedy + iterative bit-flipping
**Limitation**: No mechanism to identify the correct solution among valid alternatives

#### 2. Solution Determinism
**Challenge**: Need deterministic reconstruction to recover original data
**Proposed Solution**: Track solution indices (0=base, 1, 2, 3...) for each cube
**Implementation Status**: Not yet implemented

#### 3. Bit-Flipping Patterns
**Issue**: Internal cube modifications can maintain valid exterior sums
**Example**: Swapping bits within the same row/column/layer preserves axis sums
**Impact**: Reconstruction may produce valid but incorrect internal structure

#### 4. Scalability Considerations
**Current**: 64³ cubes = 262,144 bytes per chunk
**Memory**: Each cube requires ~1MB RAM during processing
**Performance**: O(n³) complexity for reconstruction iterations

## Implementation Roadmap

### Phase 1: Foundation (Current Status)
- [x] Basic 3D cube chunking and axis sum calculation
- [x] ZIP-based compression with LZMA
- [x] Greedy reconstruction algorithm
- [x] HTML visualization for 4³ cubes
- [x] Basic validation framework

### Phase 2: Algorithm Refinement
- [ ] **Solution Indexing System**
  - Implement pattern-based solution enumeration
  - Track solution indices during compression
  - Use indices for deterministic reconstruction

- [ ] **Enhanced Reconstruction**
  - Implement constraint propagation techniques
  - Add backtracking for complex cases
  - Optimize bit-flipping strategy

- [ ] **Validation Framework**
  - Comprehensive unit tests for all cube sizes
  - Bit-perfect reconstruction verification
  - Performance benchmarking suite

### Phase 3: Optimization
- [ ] **Multi-dimensional Constraints**
  - Add diagonal sum projections
  - Implement cross-sectional validation
  - Explore higher-order moment constraints

- [ ] **Performance Enhancements**
  - GPU acceleration for large cubes
  - Parallel processing for multiple cubes
  - Memory optimization for streaming

- [ ] **Adaptive Cube Sizing**
  - Dynamic cube size selection based on data characteristics
  - Hybrid approaches for different data types
  - Compression ratio optimization

### Phase 4: Advanced Features
- [ ] **Error Correction**
  - Redundancy mechanisms for lossy transmission
  - Checksum validation
  - Partial reconstruction capabilities

- [ ] **Specialized Formats**
  - Video-optimized compression (KBXV)
  - Audio compression (KBXA)
  - Image compression (KBXI)
  - Real-time streaming support

## Testing Strategy

### Current Testing Infrastructure

**Unit Tests (`Test/unit_tests.py`)**
- Validates axis sum consistency after reconstruction
- Tracks invalid locations and reconstruction accuracy
- Provides detailed debugging output

**HTML Visualization (`genHTML.py`)**
- Generates interactive 3D puzzle representations
- Shows axis sums and cube structure
- Enables manual verification of small cubes (4³)

**Performance Benchmarking**
- Python 3.4.3: 39.3s compression, 437.7s decompression (1MB file)
- Python 2.7.9: 39.6s compression, 41.9s decompression (10.5x faster)
- PyPy 2.5: 15.0s compression, 15.2s decompression (23x faster)

### Proposed Testing Enhancements

**1. Comprehensive Validation Suite**
```python
def test_reconstruction_accuracy():
    """Test bit-perfect reconstruction for various cube sizes"""
    for cube_size in [4, 8, 16, 32]:
        # Test with random data, structured data, sparse data

def test_compression_ratios():
    """Analyze compression effectiveness across file types"""
    # Video files, text files, binary executables, random data

def test_solution_uniqueness():
    """Identify cases with multiple valid solutions"""
    # Generate cubes with known solution counts
```

**2. Stress Testing**
- Large file processing (>1GB)
- Memory usage profiling
- Edge cases (all zeros, all ones, alternating patterns)

**3. Comparative Analysis**
- Benchmark against standard compression algorithms (gzip, bzip2, LZMA)
- Analyze compression ratios by data type
- Performance scaling with cube size

## Future Enhancements

### Diagonal Projections
Add diagonal sum constraints to reduce solution ambiguity:
- Face diagonals (4 per cube face, 24 total)
- Space diagonals (4 main diagonals through cube center)
- Edge diagonals (12 edge-parallel diagonals)

### Advanced Constraint Satisfaction
- **Constraint Propagation**: Use axis sums to eliminate impossible bit configurations
- **Arc Consistency**: Ensure all constraints remain satisfiable during reconstruction
- **Backtracking Search**: Systematic exploration of solution space

### Machine Learning Integration
- **Pattern Recognition**: Learn common bit patterns in different data types
- **Solution Prediction**: Train models to predict correct solutions from multiple valid options
- **Adaptive Parameters**: Optimize cube size and iteration counts based on data characteristics

### Real-time Applications
- **Streaming Compression**: Process data cubes in real-time
- **Progressive Reconstruction**: Partial decompression for preview/seeking
- **Network Optimization**: Minimize bandwidth for cube transmission

## Performance Considerations

### Memory Usage
- **Current**: ~1MB RAM per 64³ cube during processing
- **Optimization**: Stream processing for large files
- **Target**: Constant memory usage regardless of file size

### Computational Complexity
- **Compression**: O(n³) per cube (linear scan)
- **Decompression**: O(k×n³) where k = iteration count
- **Bottleneck**: Iterative reconstruction algorithm

### Storage Efficiency
- **Theoretical Minimum**: 3n integers vs n³ bits
- **Practical Overhead**: ZIP compression, metadata, padding
- **Current Ratio**: Varies significantly by data type and cube size

## Conclusion

KUBIX represents a novel approach to data compression using 3D geometric constraints. While the current implementation demonstrates the core concept, significant challenges remain in ensuring deterministic reconstruction and optimizing compression ratios. The project provides a solid foundation for exploring constraint-based compression techniques and offers potential applications in specialized domains where the unique properties of 3D projection-based compression may provide advantages over traditional algorithms.

The immediate focus should be on implementing solution indexing to achieve bit-perfect reconstruction, followed by comprehensive testing and performance optimization. The long-term vision includes advanced constraint satisfaction techniques and specialized applications for multimedia compression.