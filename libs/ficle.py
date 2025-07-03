
def decode7bit(bytes):
    bytes = list(bytes)
    value, shift = 0, 0
    while True:
        byteval = ord(bytes.pop(0))
        if (byteval & 128) == 0:
            break
        value |= ((byteval & 0x7F) << shift)
        shift += 7
    return (value | (byteval << shift))


def encode7bit(value):
    temp, bytes = value, ""
    while temp >= 128:
        bytes += chr(0x000000FF & (temp | 0x80))
        temp >>= 7
    bytes += chr(temp)
    return bytes

chunkSize = 2048

# Set files and file location
inputFilename = 'Output/output'
inX = inputFilename + '.kbx'
inY = inputFilename + '.kby'
inZ = inputFilename + '.kbz'
inI = inputFilename + '.kbi'

file1 = open('Output/kubix-arch-x.txt', "w", 1048576)
file2 = open('Output/kubix-arch-y.txt', "w", 1048576)
file3 = open('Output/kubix-arch-z.txt', "w", 1048576)

with \
        open(inZ, 'r') as z, \
        open(inY, 'r') as y, \
        open(inX, 'r') as x:

    # Until end of the file is reached:
    while True:
        fileX = x.read(chunkSize)
        fileY = y.read(chunkSize)
        fileZ = z.read(chunkSize)

        # Output
        file1.write(encode7bit(int(fileX)))
        file2.write(encode7bit(int(fileY)))
        file3.write(encode7bit(int(fileZ)))

file1.close()
file2.close()
file3.close()
