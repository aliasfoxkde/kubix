# KUBIX (TM) Copyright (C) 2015. Micheal L. C. Kinney. All Rights Reserved.
# Use, copying, modification, duplication, distribution, or any action
# without the direct consent of the owner listed above is absolutely
# prohibited by law.

# Optimizations & Improvements:
# Bug fixes, Pass-through Functions, PyCUDA, MultiThreading, Parallel Processing,
# Fast Integer Compression, Configuration File loading, Pass-through logic, etc.
# https://mathema.tician.de/software/pycuda/

import binascii

# Functions
def byte_to_binary(n):
    return ''.join(str((n & (1 << i)) and 1) for i in reversed(range(8)))

def hex_to_binary(h):
    return ''.join(byte_to_binary(ord(b)) for b in binascii.unhexlify(h))

enableLogTime = True
if enableLogTime == True:
    import time
    start_time = time.time()

# Variables Set by Configuration
cube = 4
cubeDims = 3
cubeDim = cube
chunkCount = 0
chunkSize = (cubeDim**3)//8  # requires dividing by the binary conversion "8"

# Variables used for testing
charLen = str("{0:0="+str(len(str(cube)))+"d}")

# Files and file locations/variables
fullLogFilename = 'Logs/full-log-output.txt'
gameLogFilename = 'Logs/game-output.txt'
binDataDump = 'Logs/bin-data-dump.txt'
filename = 'Samples/SampleVideo_1080x720_1mb.mp4'
outputFilename = 'Output/output'
outX = outputFilename + '.kbx'
outY = outputFilename + '.kby'
outZ = outputFilename + '.kbz'
outI = outputFilename + '.kbi'

# Debugging and logging Variables
verboseMode = False
logVerbose = False
fullLogDump = False
debugOnly = False
dumpChunk = False
binDump = False
gameLogger = False

# Creates a Complete Log File of all verbose statements
if fullLogDump:
    import sys
    sys.stdout = open(fullLogFilename, "w")

# Preset variables
x, y, z = cube, cube, cube
storeSide1, storeSide2, storeSide3 = list(), list(), list()
outputArray1, outputArray2, outputArray3 = list(), list(), list()

# Reads the file one chunk at a time until EOF
def read_in_chunks(infile, chunk_size=chunkSize):
    while True:
        chunk = infile.read(chunk_size)

        if chunk:
            yield chunk
        else:
            # If chunk was empty we're at the end of the file.
            return
    return

file1 = open(outX, "a+", 1048576)
file2 = open(outY, "a+", 1048576)
file3 = open(outZ, "a+", 1048576)

with open(filename, 'rb', 1048576) as readfile:
    for chunk in read_in_chunks(readfile):
        chunkCount += 1
        data = hex_to_binary(binascii.hexlify(chunk))

        if int(len(data)) == cube ** 3:
            if dumpChunk == True and chunkCount == 1:
                file = open('chunk-debug.txt', "w", 1048576)
                file.write(chunk)
                file.close()

            if binDump == True and chunkCount == 1:
                file = open('bin-debug.txt', "w", 1048576)
                file.write(data)
                file.close()

            for z in range(cube):
                for y in range(cube):
                    for x in range(cube):

                        if gameLogger == True and x == cube and y == cube:
                            gameLogFile.write('Side 3 Row {0}: {1}\n'.format(y, outputArray3))

                        storeSide1.append(data[x*cube**2+(y+(z*cube))])
                        storeSide2.append(data[(z*cube)+((cube**2)*y)+(cube-x)-1])
                        storeSide3.append(data[x*cube+y+z*cube**2])

                        if verboseMode == True:
                            if x != cube and y != cube:
                                print('Side1: Pos {0} Value {1}'.format(((x+1)+((cube**2)*y+((cube)*z))),
                                      data[((cube+(((cube**2)*x)-y))+((cube)*z))]))
                                print('Side2: Pos {0} Value {1}'.format(((cube+(((cube**2)*x)-y))+((cube)*z)),
                                      data[((cube+(((cube**2)*x)-y))+((cube)*z))]))
                                print('Side3: Pos {0} Value {1}'.format(((1+(cube*y))+(x+(cube**2)*z)),
                                      data[((1+(cube*y))+(x+(cube**2)*z))]))
                                print('Axis X: {0}, Y: {1}, Z: {2} '.format(x, y, z))
                                print('\n')

                    # Improve by making a loop that appends the variable with the dim (less code and beyond 4D)?
                    if x != cube or y != cube:
                        outputArray1.append(charLen.format(sum(map(int, storeSide1))))
                        storeSide1 = list()

                        outputArray2.append(charLen.format(sum(map(int, storeSide2))))
                        storeSide2 = list()

                        outputArray3.append(charLen.format(sum(map(int, storeSide3))))
                        storeSide3 = list()

                if verboseMode == True:
                    print('Side 1 Values: {0}'.format(outputArray1))
                    print('Side 2 Values: {0}'.format(outputArray2))
                    print('Side 3 Values: {0}'.format(outputArray3))
                    print('\n')

                for item in outputArray1:
                    file1.write("{}".format(item))

                for item in outputArray2:
                    file2.write("{}".format(item))

                for item in outputArray3:
                    file3.write("{}".format(item))

                # Resets lists
                outputArray1, outputArray2, outputArray3 = list(), list(), list()
                storeSide1, storeSide1, storeSide1 = list(), list(), list()

# Cleanup function
print('Chunk Count: {}'.format(chunkCount))
print("Total Run Time: %f seconds" % (time.time() - start_time))
