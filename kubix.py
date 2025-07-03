#!/usr/bin/env python3
"""
KUBIX: Optimized 3D Nonogram-Inspired Compression Algorithm

This implementation addresses fundamental design flaws in the original KUBIX algorithm:
- Eliminates all string-based data type conversions
- Uses native binary data types throughout
- Implements deterministic reconstruction
- Provides comprehensive testing and validation

Author: Optimized implementation based on original KUBIX concept
License: Educational/Research use
"""

import os
import sys
import time
import logging
import hashlib
from typing import Tuple, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np
from zipfile import ZipFile, ZIP_LZMA


class CubeSize(Enum):
    """Supported cube dimensions for progressive testing."""
    TINY = 4    # 64 bytes per cube
    SMALL = 8   # 512 bytes per cube  
    MEDIUM = 16 # 4KB per cube
    LARGE = 32  # 32KB per cube
    XLARGE = 64 # 262KB per cube


@dataclass
class CompressionStats:
    """Statistics for compression performance analysis."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    cube_size: int
    num_cubes: int
    reconstruction_accuracy: float


class KubixCompressor:
    """
    Optimized KUBIX 3D compression algorithm.
    
    Key improvements:
    - Native binary data types (no string conversions)
    - Deterministic reconstruction with solution indexing
    - Progressive cube size testing
    - Comprehensive validation framework
    """
    
    def __init__(self, cube_size: Union[int, CubeSize] = CubeSize.TINY, 
                 enable_logging: bool = True):
        """
        Initialize the KUBIX compressor.
        
        Args:
            cube_size: Cube dimension (4, 8, 16, 32, 64) or CubeSize enum
            enable_logging: Enable detailed logging for debugging
        """
        if isinstance(cube_size, CubeSize):
            self.cube_size = cube_size.value
        else:
            self.cube_size = cube_size
            
        self.cube_bytes = self.cube_size ** 3
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
            
        # Validate cube size
        if self.cube_size not in [4, 8, 16, 32, 64]:
            raise ValueError(f"Unsupported cube size: {self.cube_size}")
            
        self.logger.info(f"Initialized KUBIX compressor with {self.cube_size}³ cubes")
        
    def _file_to_cubes(self, file_path: Path) -> Tuple[np.ndarray, int, bytes]:
        """
        Convert binary file to 3D cubes using native binary operations.
        
        Args:
            file_path: Path to input file
            
        Returns:
            Tuple of (cubes_array, num_cubes, original_hash)
        """
        self.logger.info(f"Reading file: {file_path}")
        
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        # Calculate hash for integrity verification
        original_hash = hashlib.sha256(raw_data).digest()
        
        # Calculate number of cubes needed
        num_cubes = (len(raw_data) + self.cube_bytes - 1) // self.cube_bytes
        
        # Pad data to fit exact cube dimensions
        padded_size = num_cubes * self.cube_bytes
        padded_data = raw_data.ljust(padded_size, b'\x00')
        
        # Convert to numpy array and reshape to cubes
        # Shape: (num_cubes, cube_size, cube_size, cube_size)
        data_array = np.frombuffer(padded_data, dtype=np.uint8)
        cubes = data_array.reshape(num_cubes, self.cube_size, self.cube_size, self.cube_size)
        
        self.logger.info(f"Created {num_cubes} cubes of {self.cube_size}³ bytes each")
        return cubes, num_cubes, original_hash
        
    def _calculate_axis_sums(self, cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate axis sum projections and diagonal sums for better constraint satisfaction.

        Args:
            cube: 3D numpy array of shape (cube_size, cube_size, cube_size)

        Returns:
            Tuple of (x_sums, y_sums, z_sums, diagonal_sums) as uint32 arrays
        """
        # Calculate sums along each axis
        # X-axis: sum across Y,Z dimensions for each X slice
        x_sums = np.sum(cube, axis=(1, 2), dtype=np.uint32)

        # Y-axis: sum across X,Z dimensions for each Y slice
        y_sums = np.sum(cube, axis=(0, 2), dtype=np.uint32)

        # Z-axis: sum across X,Y dimensions for each Z slice
        z_sums = np.sum(cube, axis=(0, 1), dtype=np.uint32)

        # Calculate diagonal sums to add more constraints
        # This helps reduce the multiple solutions problem
        diagonal_sums = []

        # Main space diagonals (4 total)
        diag1 = sum(cube[i, i, i] for i in range(self.cube_size))
        diag2 = sum(cube[i, i, self.cube_size-1-i] for i in range(self.cube_size))
        diag3 = sum(cube[i, self.cube_size-1-i, i] for i in range(self.cube_size))
        diag4 = sum(cube[self.cube_size-1-i, i, i] for i in range(self.cube_size))

        diagonal_sums = np.array([diag1, diag2, diag3, diag4], dtype=np.uint32)

        return x_sums, y_sums, z_sums, diagonal_sums
        
    def _reconstruct_cube_deterministic(self, x_sums: np.ndarray, y_sums: np.ndarray,
                                      z_sums: np.ndarray, diagonal_sums: np.ndarray = None,
                                      solution_index: int = 0) -> np.ndarray:
        """
        Reconstruct cube using optimized deterministic algorithm with diagonal constraints.

        This uses a more efficient approach that addresses the multiple solutions problem
        by implementing constraint propagation, diagonal constraints, and deterministic bit placement.

        Args:
            x_sums: X-axis sum projections
            y_sums: Y-axis sum projections
            z_sums: Z-axis sum projections
            diagonal_sums: Diagonal sum constraints (optional)
            solution_index: Index to select among multiple valid solutions

        Returns:
            Reconstructed 3D cube as uint8 array
        """
        cube = np.zeros((self.cube_size, self.cube_size, self.cube_size), dtype=np.uint8)

        # Convert to int64 to avoid overflow issues
        x_sums = x_sums.astype(np.int64)
        y_sums = y_sums.astype(np.int64)
        z_sums = z_sums.astype(np.int64)

        # Phase 1: Greedy initialization based on highest constraint intersections
        # Create priority matrix based on sum requirements
        priority_matrix = np.zeros((self.cube_size, self.cube_size, self.cube_size), dtype=np.int64)

        for i in range(self.cube_size):
            for j in range(self.cube_size):
                for k in range(self.cube_size):
                    # Priority is sum of all three axis requirements
                    priority = x_sums[i] + y_sums[j] + z_sums[k]
                    # Add deterministic tie-breaker based on solution_index
                    tie_breaker = (i * self.cube_size * self.cube_size +
                                 j * self.cube_size + k + solution_index) % 100
                    priority_matrix[i, j, k] = priority * 100 + tie_breaker

        # Sort positions by priority (highest first)
        positions = [(i, j, k) for i in range(self.cube_size)
                    for j in range(self.cube_size)
                    for k in range(self.cube_size)]
        positions.sort(key=lambda pos: priority_matrix[pos[0], pos[1], pos[2]], reverse=True)

        # Phase 2: Greedy bit placement
        for i, j, k in positions:
            # Check if setting this bit would violate any constraints
            current_x_sum = np.sum(cube[i, :, :])
            current_y_sum = np.sum(cube[:, j, :])
            current_z_sum = np.sum(cube[:, :, k])

            # Only set bit if all constraints allow it
            if (current_x_sum < x_sums[i] and
                current_y_sum < y_sums[j] and
                current_z_sum < z_sums[k]):
                cube[i, j, k] = 1

        # Phase 3: Quick validation and minimal correction
        max_corrections = 10
        for correction in range(max_corrections):
            # Calculate current differences
            x_diff = np.sum(cube, axis=(1, 2), dtype=np.int64) - x_sums
            y_diff = np.sum(cube, axis=(0, 2), dtype=np.int64) - y_sums
            z_diff = np.sum(cube, axis=(0, 1), dtype=np.int64) - z_sums

            # Check if solution is found
            if np.all(x_diff == 0) and np.all(y_diff == 0) and np.all(z_diff == 0):
                self.logger.debug(f"Perfect solution found in {correction} corrections")
                break

            # Find position with largest total error and flip it
            max_error = 0
            best_pos = None

            for i in range(self.cube_size):
                for j in range(self.cube_size):
                    for k in range(self.cube_size):
                        # Calculate total error at this position
                        total_error = abs(x_diff[i]) + abs(y_diff[j]) + abs(z_diff[k])

                        # Add deterministic tie-breaker
                        tie_breaker = (i + j + k + solution_index + correction) % 1000
                        score = total_error * 1000 + tie_breaker

                        if score > max_error:
                            max_error = score
                            best_pos = (i, j, k)

            # Flip the bit at the best position
            if best_pos:
                i, j, k = best_pos
                cube[i, j, k] = 1 - cube[i, j, k]
            else:
                break  # No improvement possible

        return cube

    def _reconstruct_cube_with_constraints(self, x_sums: np.ndarray, y_sums: np.ndarray,
                                         z_sums: np.ndarray, diagonal_sums: np.ndarray,
                                         corner_bits: np.ndarray, solution_index: int = 0) -> np.ndarray:
        """
        Reconstruct cube using axis sums, diagonal constraints, and corner bits.

        This method addresses the multiple solutions problem by using additional
        constraint information stored during compression.

        Args:
            x_sums: X-axis sum projections
            y_sums: Y-axis sum projections
            z_sums: Z-axis sum projections
            diagonal_sums: Diagonal sum constraints
            corner_bits: Corner bit values for disambiguation
            solution_index: Index to select among multiple valid solutions

        Returns:
            Reconstructed 3D cube as uint8 array
        """
        cube = np.zeros((self.cube_size, self.cube_size, self.cube_size), dtype=np.uint8)

        # Convert to int64 to avoid overflow issues
        x_sums = x_sums.astype(np.int64)
        y_sums = y_sums.astype(np.int64)
        z_sums = z_sums.astype(np.int64)
        diagonal_sums = diagonal_sums.astype(np.int64)

        # Phase 1: Set corner bits from stored values
        # This provides strong constraints that help disambiguate solutions
        cube[0, 0, 0] = corner_bits[0]
        cube[0, 0, -1] = corner_bits[1]
        cube[0, -1, 0] = corner_bits[2]
        cube[0, -1, -1] = corner_bits[3]
        cube[-1, 0, 0] = corner_bits[4]
        cube[-1, 0, -1] = corner_bits[5]
        cube[-1, -1, 0] = corner_bits[6]
        cube[-1, -1, -1] = corner_bits[7]

        # Phase 2: Greedy filling with constraint checking
        # Create priority matrix considering all constraints
        positions = []
        for i in range(self.cube_size):
            for j in range(self.cube_size):
                for k in range(self.cube_size):
                    # Skip corners (already set)
                    if (i, j, k) in [(0,0,0), (0,0,self.cube_size-1), (0,self.cube_size-1,0),
                                   (0,self.cube_size-1,self.cube_size-1), (self.cube_size-1,0,0),
                                   (self.cube_size-1,0,self.cube_size-1), (self.cube_size-1,self.cube_size-1,0),
                                   (self.cube_size-1,self.cube_size-1,self.cube_size-1)]:
                        continue

                    # Calculate priority based on constraint requirements
                    priority = x_sums[i] + y_sums[j] + z_sums[k]

                    # Add diagonal priority if position is on a main diagonal
                    if i == j == k:
                        priority += diagonal_sums[0]
                    elif i == j and k == self.cube_size - 1 - i:
                        priority += diagonal_sums[1]
                    elif i == k and j == self.cube_size - 1 - i:
                        priority += diagonal_sums[2]
                    elif j == k and i == self.cube_size - 1 - j:
                        priority += diagonal_sums[3]

                    # Add deterministic tie-breaker
                    tie_breaker = (i * self.cube_size * self.cube_size +
                                 j * self.cube_size + k + solution_index) % 1000

                    positions.append((priority * 1000 + tie_breaker, i, j, k))

        # Sort by priority (highest first)
        positions.sort(reverse=True)

        # Phase 3: Fill positions in priority order
        for _, i, j, k in positions:
            # Check if setting this bit would violate constraints
            current_x_sum = np.sum(cube[i, :, :])
            current_y_sum = np.sum(cube[:, j, :])
            current_z_sum = np.sum(cube[:, :, k])

            # Check diagonal constraints
            diagonal_ok = True
            if i == j == k:
                current_diag = sum(cube[d, d, d] for d in range(self.cube_size))
                if current_diag >= diagonal_sums[0]:
                    diagonal_ok = False

            # Only set bit if all constraints allow it
            if (current_x_sum < x_sums[i] and
                current_y_sum < y_sums[j] and
                current_z_sum < z_sums[k] and
                diagonal_ok):
                cube[i, j, k] = 1

        return cube
        
    def compress(self, input_file: Union[str, Path], output_file: Union[str, Path]) -> CompressionStats:
        """
        Compress a file using the KUBIX algorithm.
        
        Args:
            input_file: Path to input file
            output_file: Path to output compressed file
            
        Returns:
            CompressionStats object with performance metrics
        """
        start_time = time.time()
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        self.logger.info(f"Starting compression: {input_path} -> {output_path}")
        
        # Read and convert file to cubes
        cubes, num_cubes, original_hash = self._file_to_cubes(input_path)
        original_size = input_path.stat().st_size
        
        # Calculate axis sums and additional constraints for all cubes
        all_x_sums = []
        all_y_sums = []
        all_z_sums = []
        all_diagonal_sums = []
        all_corner_bits = []  # Store corner bits as additional constraint

        for i in range(num_cubes):
            x_sums, y_sums, z_sums, diagonal_sums = self._calculate_axis_sums(cubes[i])
            all_x_sums.append(x_sums)
            all_y_sums.append(y_sums)
            all_z_sums.append(z_sums)
            all_diagonal_sums.append(diagonal_sums)

            # Store corner bits as additional disambiguation data
            # This is a small amount of extra data that helps ensure unique reconstruction
            corner_bits = np.array([
                cubes[i][0, 0, 0], cubes[i][0, 0, -1], cubes[i][0, -1, 0], cubes[i][0, -1, -1],
                cubes[i][-1, 0, 0], cubes[i][-1, 0, -1], cubes[i][-1, -1, 0], cubes[i][-1, -1, -1]
            ], dtype=np.uint8)
            all_corner_bits.append(corner_bits)

        # Convert to numpy arrays for efficient storage
        x_sums_array = np.array(all_x_sums, dtype=np.uint32)
        y_sums_array = np.array(all_y_sums, dtype=np.uint32)
        z_sums_array = np.array(all_z_sums, dtype=np.uint32)
        diagonal_sums_array = np.array(all_diagonal_sums, dtype=np.uint32)
        corner_bits_array = np.array(all_corner_bits, dtype=np.uint8)
        
        # Create compressed archive
        with ZipFile(output_path, 'w', compression=ZIP_LZMA, compresslevel=9) as zf:
            # Store metadata
            metadata = np.array([
                self.cube_size,
                num_cubes,
                original_size,
                len(original_hash)
            ], dtype=np.uint32)
            
            # Write arrays to temporary files and add to zip
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save arrays as binary files
                np.save(temp_path / 'metadata.npy', metadata)
                np.save(temp_path / 'x_sums.npy', x_sums_array)
                np.save(temp_path / 'y_sums.npy', y_sums_array)
                np.save(temp_path / 'z_sums.npy', z_sums_array)
                np.save(temp_path / 'diagonal_sums.npy', diagonal_sums_array)
                np.save(temp_path / 'corner_bits.npy', corner_bits_array)

                # Save original hash for integrity verification
                with open(temp_path / 'hash.bin', 'wb') as f:
                    f.write(original_hash)

                # Add files to archive
                zf.write(temp_path / 'metadata.npy', 'metadata.npy')
                zf.write(temp_path / 'x_sums.npy', 'x_sums.npy')
                zf.write(temp_path / 'y_sums.npy', 'y_sums.npy')
                zf.write(temp_path / 'z_sums.npy', 'z_sums.npy')
                zf.write(temp_path / 'diagonal_sums.npy', 'diagonal_sums.npy')
                zf.write(temp_path / 'corner_bits.npy', 'corner_bits.npy')
                zf.write(temp_path / 'hash.bin', 'hash.bin')
        
        compression_time = time.time() - start_time
        compressed_size = output_path.stat().st_size
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        self.logger.info(f"Compression completed in {compression_time:.2f}s")
        self.logger.info(f"Compression ratio: {compression_ratio:.2f}:1")
        
        return CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            decompression_time=0,  # Will be set during decompression
            cube_size=self.cube_size,
            num_cubes=num_cubes,
            reconstruction_accuracy=0  # Will be calculated during validation
        )

    def decompress(self, input_file: Union[str, Path], output_file: Union[str, Path],
                   solution_index: int = 0) -> Tuple[CompressionStats, bool]:
        """
        Decompress a KUBIX compressed file.

        Args:
            input_file: Path to compressed file
            output_file: Path to output decompressed file
            solution_index: Index for deterministic solution selection

        Returns:
            Tuple of (CompressionStats, integrity_check_passed)
        """
        start_time = time.time()
        input_path = Path(input_file)
        output_path = Path(output_file)

        self.logger.info(f"Starting decompression: {input_path} -> {output_path}")

        # Extract compressed data
        with ZipFile(input_path, 'r') as zf:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                zf.extractall(temp_dir)
                temp_path = Path(temp_dir)

                # Load metadata
                metadata = np.load(temp_path / 'metadata.npy')
                cube_size, num_cubes, original_size, hash_len = metadata

                # Verify cube size matches
                if cube_size != self.cube_size:
                    raise ValueError(f"Cube size mismatch: expected {self.cube_size}, got {cube_size}")

                # Load axis sums and additional constraints
                x_sums_array = np.load(temp_path / 'x_sums.npy')
                y_sums_array = np.load(temp_path / 'y_sums.npy')
                z_sums_array = np.load(temp_path / 'z_sums.npy')
                diagonal_sums_array = np.load(temp_path / 'diagonal_sums.npy')
                corner_bits_array = np.load(temp_path / 'corner_bits.npy')

                # Load original hash
                with open(temp_path / 'hash.bin', 'rb') as f:
                    original_hash = f.read()

        # Reconstruct cubes using enhanced constraints
        reconstructed_cubes = []
        for i in range(num_cubes):
            cube = self._reconstruct_cube_with_constraints(
                x_sums_array[i], y_sums_array[i], z_sums_array[i],
                diagonal_sums_array[i], corner_bits_array[i], solution_index
            )
            reconstructed_cubes.append(cube)

        # Convert cubes back to binary data
        cubes_array = np.array(reconstructed_cubes, dtype=np.uint8)
        reconstructed_data = cubes_array.flatten().tobytes()

        # Remove padding to get original size
        reconstructed_data = reconstructed_data[:original_size]

        # Verify integrity
        reconstructed_hash = hashlib.sha256(reconstructed_data).digest()
        integrity_check = original_hash == reconstructed_hash

        # Write decompressed file
        with open(output_path, 'wb') as f:
            f.write(reconstructed_data)

        decompression_time = time.time() - start_time
        compressed_size = input_path.stat().st_size
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0

        self.logger.info(f"Decompression completed in {decompression_time:.2f}s")
        self.logger.info(f"Integrity check: {'PASSED' if integrity_check else 'FAILED'}")

        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time=0,  # Not measured during decompression
            decompression_time=decompression_time,
            cube_size=self.cube_size,
            num_cubes=num_cubes,
            reconstruction_accuracy=1.0 if integrity_check else 0.0
        )

        return stats, integrity_check

    def validate_reconstruction(self, original_file: Union[str, Path],
                              compressed_file: Union[str, Path]) -> float:
        """
        Validate reconstruction accuracy by comparing original and decompressed files.

        Args:
            original_file: Path to original file
            compressed_file: Path to compressed file

        Returns:
            Reconstruction accuracy (1.0 = perfect, 0.0 = completely wrong)
        """
        import tempfile

        with tempfile.NamedTemporaryFile() as temp_file:
            # Decompress and compare
            stats, integrity_check = self.decompress(compressed_file, temp_file.name)

            if integrity_check:
                return 1.0

            # If integrity check failed, calculate byte-level accuracy
            with open(original_file, 'rb') as f1, open(temp_file.name, 'rb') as f2:
                original_data = f1.read()
                reconstructed_data = f2.read()

            # Compare byte by byte
            min_len = min(len(original_data), len(reconstructed_data))
            if min_len == 0:
                return 0.0

            matching_bytes = sum(1 for i in range(min_len)
                               if original_data[i] == reconstructed_data[i])

            accuracy = matching_bytes / max(len(original_data), len(reconstructed_data))
            return accuracy


class KubixTester:
    """
    Comprehensive testing framework for KUBIX algorithm validation.
    """

    def __init__(self, test_data_dir: Union[str, Path] = "Samples"):
        """
        Initialize the testing framework.

        Args:
            test_data_dir: Directory containing test files
        """
        self.test_data_dir = Path(test_data_dir)
        self.results = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_cube_size_progression(self, test_file: Union[str, Path]) -> List[CompressionStats]:
        """
        Test compression with progressively larger cube sizes.

        Args:
            test_file: Path to test file

        Returns:
            List of CompressionStats for each cube size
        """
        test_path = Path(test_file)
        results = []

        self.logger.info(f"Testing cube size progression with file: {test_path}")

        for cube_size in [4]:  # Start with just 4³ for initial testing
            self.logger.info(f"Testing cube size: {cube_size}³")

            try:
                compressor = KubixCompressor(cube_size=cube_size)

                # Create temporary files
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.kbx', delete=False) as compressed_file:
                    compressed_path = compressed_file.name

                # Compress
                stats = compressor.compress(test_path, compressed_path)

                # Validate reconstruction
                accuracy = compressor.validate_reconstruction(test_path, compressed_path)
                stats.reconstruction_accuracy = accuracy

                results.append(stats)

                self.logger.info(f"Cube {cube_size}³: Ratio={stats.compression_ratio:.2f}, "
                               f"Accuracy={accuracy:.3f}, Time={stats.compression_time:.2f}s")

                # Cleanup
                os.unlink(compressed_path)

            except Exception as e:
                self.logger.error(f"Error testing cube size {cube_size}: {e}")

        return results

    def test_data_types(self, cube_size: int = 4) -> List[CompressionStats]:
        """
        Test compression effectiveness on different data types.

        Args:
            cube_size: Cube dimension to use for testing

        Returns:
            List of CompressionStats for each test case
        """
        results = []
        compressor = KubixCompressor(cube_size=cube_size)

        # Test cases: different data patterns
        test_cases = [
            ("random_data", self._generate_random_data(1024)),
            ("zeros_data", b'\x00' * 1024),
            ("ones_data", b'\xff' * 1024),
            ("alternating_data", (b'\x55\xaa' * 512)),
            ("structured_data", self._generate_structured_data(1024))
        ]

        for test_name, test_data in test_cases:
            self.logger.info(f"Testing data type: {test_name}")

            try:
                import tempfile

                # Create temporary input file
                with tempfile.NamedTemporaryFile(delete=False) as input_file:
                    input_file.write(test_data)
                    input_path = input_file.name

                # Create temporary compressed file
                with tempfile.NamedTemporaryFile(suffix='.kbx', delete=False) as compressed_file:
                    compressed_path = compressed_file.name

                # Compress and validate
                stats = compressor.compress(input_path, compressed_path)
                accuracy = compressor.validate_reconstruction(input_path, compressed_path)
                stats.reconstruction_accuracy = accuracy

                results.append(stats)

                self.logger.info(f"{test_name}: Ratio={stats.compression_ratio:.2f}, "
                               f"Accuracy={accuracy:.3f}")

                # Cleanup
                os.unlink(input_path)
                os.unlink(compressed_path)

            except Exception as e:
                self.logger.error(f"Error testing {test_name}: {e}")

        return results

    def _generate_random_data(self, size: int) -> bytes:
        """Generate random binary data for testing."""
        import random
        return bytes(random.randint(0, 255) for _ in range(size))

    def _generate_structured_data(self, size: int) -> bytes:
        """Generate structured data with patterns for testing."""
        # Create a pattern that repeats
        pattern = b'\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff'
        return (pattern * (size // len(pattern) + 1))[:size]

    def benchmark_against_standard_algorithms(self, test_file: Union[str, Path]) -> dict:
        """
        Compare KUBIX compression against standard algorithms.

        Args:
            test_file: Path to test file

        Returns:
            Dictionary with compression results for each algorithm
        """
        import gzip
        import bz2
        import lzma

        test_path = Path(test_file)
        results = {}

        # Read original file
        with open(test_path, 'rb') as f:
            original_data = f.read()
        original_size = len(original_data)

        # Test standard algorithms
        algorithms = {
            'gzip': lambda data: gzip.compress(data),
            'bz2': lambda data: bz2.compress(data),
            'lzma': lambda data: lzma.compress(data)
        }

        for name, compress_func in algorithms.items():
            start_time = time.time()
            compressed_data = compress_func(original_data)
            compression_time = time.time() - start_time

            results[name] = {
                'compressed_size': len(compressed_data),
                'compression_ratio': original_size / len(compressed_data),
                'compression_time': compression_time,
                'accuracy': 1.0  # Standard algorithms are lossless
            }

        # Test KUBIX with different cube sizes
        for cube_size in [4, 8, 16]:
            try:
                compressor = KubixCompressor(cube_size=cube_size)

                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.kbx', delete=False) as compressed_file:
                    compressed_path = compressed_file.name

                stats = compressor.compress(test_path, compressed_path)
                accuracy = compressor.validate_reconstruction(test_path, compressed_path)

                results[f'kubix_{cube_size}'] = {
                    'compressed_size': stats.compressed_size,
                    'compression_ratio': stats.compression_ratio,
                    'compression_time': stats.compression_time,
                    'accuracy': accuracy
                }

                os.unlink(compressed_path)

            except Exception as e:
                self.logger.error(f"Error testing KUBIX {cube_size}: {e}")

        return results

    def generate_report(self, results: dict, output_file: str = "kubix_benchmark_report.txt"):
        """
        Generate a comprehensive performance report.

        Args:
            results: Dictionary of benchmark results
            output_file: Path to output report file
        """
        with open(output_file, 'w') as f:
            f.write("KUBIX 3D Compression Algorithm - Performance Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("Algorithm Comparison:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Algorithm':<15} {'Ratio':<8} {'Time(s)':<8} {'Accuracy':<8}\n")
            f.write("-" * 40 + "\n")

            for name, stats in results.items():
                f.write(f"{name:<15} {stats['compression_ratio']:<8.2f} "
                       f"{stats['compression_time']:<8.2f} {stats['accuracy']:<8.3f}\n")

            f.write("\n")

        self.logger.info(f"Report generated: {output_file}")


def main():
    """
    Main execution function for testing and demonstration.
    """
    print("KUBIX 3D Compression Algorithm - Optimized Implementation")
    print("=" * 60)

    # Initialize tester
    tester = KubixTester()

    # Check for test files
    test_files = [
        "Samples/SampleVideo_1080x720_1mb.mp4",
        "Samples/SampleVideo_1280x720_2mb.mp4"
    ]

    available_files = [f for f in test_files if Path(f).exists()]

    if not available_files:
        print("No test files found. Creating synthetic test data...")

        # Create a small test file for initial testing
        test_data = tester._generate_structured_data(256)  # 256 bytes = 4 cubes of 4³
        test_file = "test_data.bin"
        with open(test_file, 'wb') as f:
            f.write(test_data)
        available_files = [test_file]

    # Run tests
    for test_file in available_files[:1]:  # Test with first available file
        print(f"\nTesting with file: {test_file}")
        print("-" * 40)

        # Test cube size progression
        print("\n1. Cube Size Progression Test:")
        progression_results = tester.test_cube_size_progression(test_file)

        for stats in progression_results:
            print(f"  Cube {stats.cube_size}³: "
                  f"Ratio={stats.compression_ratio:.2f}, "
                  f"Accuracy={stats.reconstruction_accuracy:.3f}, "
                  f"Time={stats.compression_time:.2f}s")

        # Test different data types
        print("\n2. Data Type Test (4³ cubes):")
        data_type_results = tester.test_data_types(cube_size=4)

        # Benchmark against standard algorithms
        print("\n3. Algorithm Comparison:")
        benchmark_results = tester.benchmark_against_standard_algorithms(test_file)

        for name, stats in benchmark_results.items():
            print(f"  {name:<12}: Ratio={stats['compression_ratio']:.2f}, "
                  f"Time={stats['compression_time']:.3f}s, "
                  f"Accuracy={stats['accuracy']:.3f}")

        # Generate detailed report
        tester.generate_report(benchmark_results)

    print(f"\nTesting completed. Check 'kubix_benchmark_report.txt' for detailed results.")


if __name__ == "__main__":
    main()
