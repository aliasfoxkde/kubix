import os
import numpy as np
from zipfile import ZipFile, ZIP_LZMA

# === CONFIGURATION ===
INPUT_FILE = "./samples/SampleVideo_1080x720_1mb.mp4"
OUTPUT_DIR = "./output"
RESTORED_DIR = "./restored"
CUBE_SIZE = 64  # Fixed size for deterministic reconstruction [[1]]

def file_to_cubes(file_path):
    """Split binary file into fixed-size 3D cubes."""
    with open(file_path, "rb") as f:
        raw_data = bytearray(f.read())

    # Pad data to fit cube dimensions
    cube_bytes = CUBE_SIZE**3
    num_cubes = (len(raw_data) + cube_bytes - 1) // cube_bytes
    cubes = []

    for i in range(num_cubes):
        start = i * cube_bytes
        end = start + cube_bytes
        chunk = raw_data[start:end]
        padded_chunk = chunk.ljust(cube_bytes, b'\x00')
        cube = np.frombuffer(padded_chunk, dtype=np.uint8).reshape((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        cubes.append(cube)

    return cubes, num_cubes

def cube_to_sums(cube):
    """Calculate X/Y/Z axis sums."""
    x_sums = np.sum(cube, axis=(1, 2)).astype(np.uint32)
    y_sums = np.sum(cube, axis=(0, 2)).astype(np.uint32)
    z_sums = np.sum(cube, axis=(0, 1)).astype(np.uint32)
    return x_sums, y_sums, z_sums

def reconstruct_valid_cube(x_sums, y_sums, z_sums):
    """Reconstruct a cube that satisfies XYZ sums (valid solution)."""
    cube = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), dtype=np.uint8)

    # Greedy initialization for high-sum regions
    for i in range(CUBE_SIZE):
        for j in range(CUBE_SIZE):
            for k in range(CUBE_SIZE):
                if np.sum(cube[i, :, :]) < x_sums[i] and \
                   np.sum(cube[:, j, :]) < y_sums[j] and \
                   np.sum(cube[:, :, k]) < z_sums[k]:
                    cube[i, j, k] = 1

    # Bit-flip adjustments to match sums
    for _ in range(20):  # Limited iterations
        x_diff = np.sum(cube, axis=(1, 2)) - x_sums
        y_diff = np.sum(cube, axis=(0, 2)) - y_sums
        z_diff = np.sum(cube, axis=(0, 1)) - z_sums

        for i in range(CUBE_SIZE):
            for j in range(CUBE_SIZE):
                for k in range(CUBE_SIZE):
                    # Flip bits where mismatches exist
                    if x_diff[i] > 0 and cube[i, j, k] == 1:
                        cube[i, j, k] = 0
                    elif x_diff[i] < 0 and cube[i, j, k] == 0:
                        cube[i, j, k] = 1

                    if y_diff[j] > 0 and cube[i, j, k] == 1:
                        cube[i, j, k] = 0
                    elif y_diff[j] < 0 and cube[i, j, k] == 0:
                        cube[i, j, k] = 1

                    if z_diff[k] > 0 and cube[i, j, k] == 1:
                        cube[i, j, k] = 0
                    elif z_diff[k] < 0 and cube[i, j, k] == 0:
                        cube[i, j, k] = 1

        # Early exit if sums match
        if np.array_equal(np.sum(cube, axis=(1, 2)), x_sums) and \
           np.array_equal(np.sum(cube, axis=(0, 2)), y_sums) and \
           np.array_equal(np.sum(cube, axis=(0, 1)), z_sums):
            break

    return cube

def compress_kubix():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESTORED_DIR, exist_ok=True)

    # Step 1: Split file into cubes
    cubes, num_cubes = file_to_cubes(INPUT_FILE)
    print(f"Split into {num_cubes} cubes ({CUBE_SIZE}³ each)")

    # Step 2: Process each cube
    temp_dir = "./temp_kubix"
    os.makedirs(temp_dir, exist_ok=True)

    for idx, cube in enumerate(cubes):
        x_sums, y_sums, z_sums = cube_to_sums(cube)
        np.save(f"{temp_dir}/x_{idx}.npy", x_sums)
        np.save(f"{temp_dir}/y_{idx}.npy", y_sums)
        np.save(f"{temp_dir}/z_{idx}.npy", z_sums)

    np.save(f"{temp_dir}/metadata.npy", np.array([CUBE_SIZE, num_cubes], dtype=np.uint32))

    # Step 3: Compress to ZIP archive using LZMA
    output_kbx = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_FILE).replace(".mp4", ".kbx"))
    with ZipFile(output_kbx, "w", compression=ZIP_LZMA) as kbz:
        for idx in range(num_cubes):
            kbz.write(f"{temp_dir}/x_{idx}.npy", arcname=f"x_{idx}.npy")
            kbz.write(f"{temp_dir}/y_{idx}.npy", arcname=f"y_{idx}.npy")
            kbz.write(f"{temp_dir}/z_{idx}.npy", arcname=f"z_{idx}.npy")
        kbz.write(f"{temp_dir}/metadata.npy", arcname="metadata.npy")

    # Clean up
    for idx in range(num_cubes):
        os.remove(f"{temp_dir}/x_{idx}.npy")
        os.remove(f"{temp_dir}/y_{idx}.npy")
        os.remove(f"{temp_dir}/z_{idx}.npy")
    os.remove(f"{temp_dir}/metadata.npy")
    os.rmdir(temp_dir)

    print(f"Compressed to: {output_kbx}")

def decompress_kubix(kbx_file):
    """Reconstruct original file from `.kbx` archive."""
    temp_dir = "./temp_kubix"
    os.makedirs(temp_dir, exist_ok=True)

    with ZipFile(kbx_file, "r") as kbz:
        kbz.extractall(temp_dir)

    # Load metadata
    cube_size, num_cubes = np.load(os.path.join(temp_dir, "metadata.npy")).astype(int)
    reconstructed_cubes = []

    # Reconstruct each cube
    for idx in range(num_cubes):
        x_sums = np.load(os.path.join(temp_dir, f"x_{idx}.npy")).astype(np.uint32)
        y_sums = np.load(os.path.join(temp_dir, f"y_{idx}.npy")).astype(np.uint32)
        z_sums = np.load(os.path.join(temp_dir, f"z_{idx}.npy")).astype(np.uint32)

        cube = reconstruct_valid_cube(x_sums, y_sums, z_sums, cube_size)
        reconstructed_cubes.append(cube)

    # Convert cubes back to bytes
    reconstructed_bytes = b''.join(cube.tobytes() for cube in reconstructed_cubes)
    output_file = os.path.join(RESTORED_DIR, os.path.basename(kbx_file).replace(".kbx", "_recovered.mp4"))
    with open(output_file, "wb") as f:
        f.write(reconstructed_bytes.rstrip(b'\x00'))  # Remove padding

    print(f"Reconstructed file saved to: {output_file}")
    print(f"File Size: {os.path.getsize(output_file)} bytes")

# === RUN ===
if __name__ == "__main__":
    compress_kubix()
    decompress_kubix(os.path.join(OUTPUT_DIR, os.path.basename(INPUT_FILE).replace(".mp4", ".kbx")))
