# kubix
This project implements a 3D compression algorithm that transforms binary data into a series of 3D cubes, stores XYZ axis sums, and reconstructs the original data by solving for a valid cube configuration that matches the sums. The system uses chunked processing to handle large files and ensures XYZ sum validation for deterministic reconstruction.
