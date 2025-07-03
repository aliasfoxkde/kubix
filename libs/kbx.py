# Custom Code Snippets for Python 2.7/Pypy
# Notes:


__author__ = 'mkinney'

def code():
    return

'''
# Optimized KBX Code Backup
# Populates an output array from input Data; logic support nth Dimensions; move to kbx_lib;
def kbx_populate(data, cube, dims, tmpArray):                                 # Populates initial reconstruct array
    o, list_size, data_size = list(), cube**(dims-1), cube**dims    # Assign dynamic variables
    xPattern, zPattern = list(), list()
    for i in range(0, list_size, 1):                                # Loop i through list
        p = int(str(data[i:list_size:data_size]))                   # Generates Pattern
        if p > cube: return "Invalid input"                         # Basic Error Checking
        o.append([1] * p + [0] * abs(cube-p))                       # Appends Output (o)
    return o, xPattern, zPattern                                    # Returns Output as o
'''

def kbx_populate(data, cube, dims):                       # Populates initial reconstruct array
    """ Populates an output array from input Data as well as XYZ Patterns (in order to reuse loop)
        Also, the logic support nth Dimensions; test and move to kbx_lib. """

    list_size, data_size = cube**(dims-1), cube**dims               # Assign dynamic variables
    o, xPattern, zPattern = list(), list(), list()                  # Initialize lists
    tmpArray = [a for a in xrange(data_size)]                       # Generates tmpArray used for 'Patterns'

    for i in range(0, list_size, 1):                                # Loop i through list_size
        p = int(str(data[i:list_size:data_size]))                   # Generates Pattern
        if p > cube: return "Invalid input"                         # Basic Error Checking; replace with Try/Catch
        o.append([1] * p + [0] * abs(cube-p))                       # Appends Output (o) (Requires Return Join())
        # o += [1] * p + [0] * abs(cube-p)                          # Concatenates Output (o) (Slower than append)

        # Generates Dynamic YXZ Patterns
        xPattern.append(tmpArray[i:data_size:list_size])
        zPattern.append(tmpArray[i+((i/cube)*(list_size-cube)):i+((i/cube)*(list_size-cube))+list_size:cube])
        # yPattern.append(tmpArray[((cube+(i*cube**2))-1)+((i/cube)*cube):((cube+(i*cube**2))-1)+((i/cube)*cube)-cube:-1])
    return o, xPattern, zPattern                                    # Returns Output as o

# Build Classes for Shift, Switch, Collisions, Performance Logger, Unit Tests, Visualization, and Validation...


def pause():
    return raw_input('Press any key to continue...')


def sqrt(x):
    return x**.5


def int2bin(value):
    if value == 0:
        return "0"
    else:
        s = ''
    while value:
        if value & 1 == 1:
            s = "1" + s
        else:
            s = "0" + s
        value /= 2
    while len(s) < 8:
        s = "0" + s
    return s
