# KUBIX (TM) Copyright (C) 2015. Micheal L. C. Kinney. All Rights Reserved.
# Use, copying, modification, duplication, distribution, or any action
# without the direct consent of the owner listed above is absolutely
# prohibited by law.

# Configure values
cube = 4
cubeDims = 3
inputFilename = 'Output/output'
enableLogTime = True

# Debugging and Unit test variables
unitTest_Validate = False
unitTest_Verbose = False
genHTML_Output = False
genHTML_verbose = False
debugging = False
haultKubix = False
haultValue = 1

# Calculates runtime if enabled
if enableLogTime:
    import time
    start_time = time.time()

# Variables Set by Configuration
baseCheck, xCheck, yCheck, zCheck, cCheck, validateCheck = 0, 0, 0, 0, 0, 0
currentZ, maxRow, maxColumn, maxNum = list(), 0, 0, cube ** cubeDims
chunkSize = cube**cubeDims-1  # chunkSize changes from 'kubix' due to 'already' being binary and only solving 'one side'
arraySize = cube**cubeDims
multipleErrorCheck = 0
charLen = str("{0:0="+str(len(str(cube)))+"d}")

# Set files and file location
inX = inputFilename + '.kbx'
inY = inputFilename + '.kby'
inZ = inputFilename + '.kbz'
inI = inputFilename + '.kbi'

# Local Variables and arrays
validChunks, chunkCountX, chunkCountY = 0, 0, 0
iArray = [0] * (cube ** (cubeDims-1))
emptyArray = [0] * (cube ** cubeDims)
compareLocs = list()

# "Un-archive Files"
#   Logic required to Read/Extract files from from zip archive/container

with \
        open(inZ, 'rb') as z, \
        open(inY, 'rb') as y, \
        open(inX, 'rb') as x:

    # Until end of the file is reached:
    while True:
        fileX = x.read(chunkSize)
        fileY = y.read(chunkSize)
        fileZ = z.read(chunkSize)
        validateSumX = sum(map(int, fileX))
        validateSumY = sum(map(int, fileY))
        validateSumZ = sum(map(int, fileZ))

        # Exits loop at the end of last file
        if not fileZ:
            print('The file has been reconstructed!\n')
            break

        # Validation logic for later used for "Transmission" for all three chunks (x,y,z)
        if validateSumZ == validateSumY == validateSumX:

            # Verbose Debugging if enabled
            if debugging::
                print('ValidatedSums', validateSumX, validateSumY, validateSumZ)

            # Preset variables
            xCount, yCount, zCount = 0, 0, 0  # Row, Column, Layer
            outputArrayX = emptyArray[:]

            # "Populate Values"
            # Simple Function: 1) Populates the outputArrayX
            for charX in fileX:
                for placement in range(int(charX), -1, -1):

                    # Placement for outputArrayX
                    if placement != 0:
                        location = yCount + (xCount * (cube ** (cubeDims-1))) + (zCount * cube)
                        if location >= cube**cubeDims:
                            break
                        outputArrayX[location] = 1
                        xCount += 1
                    elif placement == 0 and yCount == cube:
                        xCount = 0
                    elif placement == 0:
                        xCount = 0
                        yCount += 1

                    # X pattern
                    if xCount == cube+1:
                        xCount = 0
                        yCount += 1

                    if yCount == cube:
                        yCount = 0
                        zCount += 1

                    if zCount == cube:
                        zCount = 0
                        chunkCountX += 1
                        break

            outputArrayY = emptyArray[:]

            """ !!! Below doesn't work as initially intended; only really counts.
                This is probably contributing to the bug as well  !!! """

            #'''
            #  "Placement Logic"
            # Complex Function: 1) counts through charY to determine 'placement', 2) Populates the arrayY,
            # 3) shifts values until arrayX == arrayY, and 4) assigns the checkArray to the resulting TRUE value.
            for charY in fileY:
                for placement in range(int(charY), -1, -1):

                    # Placement for outputArrayY
                    if placement != 0:
                        location = (((cube**(cubeDims-1))*yCount)+cube)+(zCount*cube)-xCount-1
                        if location >= cube**cubeDims:
                            break
                        outputArrayY[location] = 1
                        xCount += 1
                    elif placement == 0 and yCount == cube:
                        xCount = 0
                    elif placement == 0:
                        xCount = 0
                        yCount += 1

                    # Missing X Counter?
                    # xCount += 1

                    # x pattern
                    if xCount == cube+1:
                        xCount = 0
                        yCount += 1

                    if yCount == cube:
                        yCount = 0
                        zCount += 1

                    if zCount == cube:
                        zCount = 0
                        chunkCountY += 1

            # Keeps a track of the valid chunks processed
            if sum(outputArrayY) == sum(outputArrayX):
                validChunks += 1

            # "Shift Logic"
            # Complex Function: 1) "Shifts" values until arrayX == arrayY, 2) assigns the checkArray to the resulting
            # TRUE value, 3) validates outputArrayY with outputArrayX and 4) if they equal each other, store to final
            # array and continue.

            # Presets Variables
            locationY, shiftUP = 0, 0
            compareSums = list()

            for charY in fileY:
                # for charX in fileX:

                for getSumLocation in range(int(cube-1), -1, -1):

                    # Modifying Effects output array
                    locationY = ((cube-getSumLocation-1)+((cube*cube)*xCount)+yCount*cube)

                    # Confirmed!!! The bug has to be here!
                    # BUG: There's a bug in the code causing the XY totals to be wrong about 17%+ of the time
                    #      and it is most likely in this section.

                    # Evaluates current.row(x) sum:
                    for temp in range(int(cube-1), -1, -1):
                        getTempPositions = ((cube-temp-1)+((cube**2)*xCount)+yCount*cube)
                        compareSums.append(outputArrayX[getTempPositions])
                        compareLocs.append(getTempPositions)

                    while sum(compareSums) != int(charY):

                        # To improve performance, could check is sum of Y/X is wrong first and only
                        #   search row if the condition is matched. Could also improve with list comprehension:
                        #   example: list[cube:cube**2:-1], or something like that

                        # Increment Shift unless specific conditions met
                        if shiftUP == cube-1 or outputArrayX[locationY] == 0:
                            shiftUP = 0
                            break
                        else:
                            shiftUP += 1

                            # Validation Check:
                            moveShift = locationY+((cube**2)*shiftUP)

                            if moveShift <= ((cube**3)-1):

                                if outputArrayX[moveShift] == 0 and outputArrayX[locationY] == 1:
                                    outputArrayX[moveShift] = 1
                                    outputArrayX[locationY] = 0

                    # Reset variables
                    compareSums = list()
                    compareLocs = list()

                # x pattern
                xCount += 1

                if xCount == cube:
                    xCount = 0
                    yCount += 1
                elif yCount == cube:
                    yCount = 0
                    zCount += 1
                elif zCount == cube:
                    zCount = 0
                    break

            # Resets variables and lists
            xCount, yCount, zCount = 0, 0, 0  # Row, Column, Layer
            validArray = outputArrayX[:]  # Sets validArray to x output array
            compareLocations = list()
            compareLocCheck = list()
            debugValues = list()
            outList = list()
            shiftUP = 0

            # "Switch Logic"
            # Complex Function: 1) Validates checkArray per "layer" (total) against fileZ (side totals), and
            # 2) "Switches" positions until values are correct.

            for charZ in fileZ:
                # Fix compareLocation logic (make while loop)
                # while sum(compareLocations) != int(charZ):
                for test in range(0, cube):
                    location = yCount + (xCount * (cube ** (cubeDims-1))) + (zCount * cube)

                    # Logic speedup: should only "switch" the first valid value
                    # and move on. This may occasionally require a recheck/fix
                    # logic but overall should be faster. Requires while logic.

                    # Note: change for 'while' condition
                    for checks in range(1, cube):
                        baseCheck = location
                        xCheck = location + (checks * (cube ** (cubeDims-1)))

                        # Note: change for 'while' condition
                        for check2 in range(1, cube):
                            yCheck = location + check2
                            validateCheck = yCheck + (checks * (cube ** (cubeDims-1)))
                            maxRow = location + ((cube - xCount - 1) * (cube ** (cubeDims-1)))
                            maxColumn = location + (cube - yCount - 1)
                            currentZ = baseCheck - (xCount * (cube * (cube - 1))) - (cube * zCount)

                            if yCheck <= maxColumn and xCheck <= maxRow:

                                if \
                                    validArray[yCheck] == validArray[xCheck] and \
                                    validArray[baseCheck] == validArray[
                                    validateCheck] and validArray[yCheck] != validArray[baseCheck] and \
                                    validArray[xCheck] != validArray[validateCheck]:

                                    # Evaluation
                                    for getSumLocation in range(cube):
                                        getPosition = yCount + (getSumLocation * cube) + (xCount * (cube ** (cubeDims-1)))
                                        compareLocations.append(validArray[getPosition])
                                        compareLocCheck.append(getPosition)

                                    if int(fileZ[currentZ]) != int(sum(compareLocations)):
                                        ''' MISSING LOGIC HERE! '''

                                        # Temporary 'hack' to solve for 4 invalid values
                                        #   Find a layer that can switch the 4 values;
                                        #   Determine if the values need to shift up, or shift down
                                        #   If the 4 values meet both criteria, shift them.
                                        #   Solves a 4 cube to 43.5%.

                                        '''
                                        # For logic reasons and debugging only
                                        if validArray[baseCheck] == 0 and int(fileZ[currentZ]) > int(sum(compareLocations)):
                                            print('\nShifting up!')

                                        if validArray[baseCheck] == 1 and int(fileZ[currentZ]) < int(sum(compareLocations)):
                                            print('\nShifting down!')
                                            '''

                                        # 'Switch' First Pair
                                        validArray[baseCheck] = validArray[yCheck]
                                        validArray[xCheck] = validArray[validateCheck]

                                        # 'Switch' Second Pair
                                        validArray[yCheck] = validArray[validateCheck]
                                        validArray[validateCheck] = validArray[baseCheck]

                            # Reset variables
                            compareLocations = list()
                            compareLocCheck = list()

                    # x pattern
                    xCount += 1

                    if xCount == cube - 1:
                        xCount = 0
                        yCount += 1

                    if yCount == cube - 1:
                        yCount = 0
                        zCount += 1

                    if zCount == cube:
                        zCount = 0
                        cCheck += 1  # 'Hack' to reduce code to 1 loop through 'cube'
                        break

                # Reset variables in loop
                compareLocations = list()
                compareLocCheck = list()
                debugValues = list()

            # !!! DEBUGGING TOOLS AND UNIT TESTS HERE !!!

            # Unit test used for validation/verification of cube/chunk
            if unitTest_Validate::
                compareLocations, compareLocCheck, invalidLocations = list(), list(), list()
                currentZ, numOfValid, numOfInvalid = 0, 0, 0
                unitStatus = 'Successful'

                for b in range(0, cube):
                    for a in range(0, cube):
                        location = a + (b * (cube ** (cubeDims-1)))

                        for getSumLocation in range(cube):
                            compareLocations.append(validArray[a + (getSumLocation * cube) + (b * (cube * cube))])
                            compareLocCheck.append(a + (getSumLocation * cube) + (b * (cube ** (cubeDims-1))))

                        if int(fileZ[currentZ]) == int(sum(compareLocations)):
                            numOfValid += 1
                        else:
                            unitStatus = 'Failed'
                            numOfInvalid += 1
                            invalidLocations.append(currentZ)

                        # Reset variables
                        compareLocations = list()
                        compareLocCheck = list()
                        currentZ += 1

                # Dynamic Output for numOfInvalid values
                for item in range(chunkSize):
                    if numOfInvalid == item:
                        iArray[item] = iArray[item] + 1

                if unitTest_Verbose::
                    output = ''.join(map(str, validArray))

                    print('\nUnit Test Summary:')
                    print('  Chunk [%s] Length [%s]' % (validChunks, len(output)))
                    print('  Array [%s]' % (output))
                    print('  Validation check %s!' % unitStatus)
                    print('  {} of {} Columns Valid ({} Invalid {})'.format(numOfValid, cube ** 2, numOfInvalid,
                                                                            invalidLocations))

                    pause = raw_input("  Press Enter to continue...")

                    # BUG: The Invalid locations appear to be off by 1 (need to shift up 1)?

            # Generates "Game" Outputs to HTML as Visualization
            if genHTML_Output::

                # Static variables
                dirPath = 'HTML_Outputs/'
                gridX, gridY, gridZ = str(fileX), str(fileY), str(fileZ)
                gridCount, popData, countZ = 0, 0, 0
                dimCount, layerCount = cubeDims, 11

                # Dynamic (changing) variables
                fileOutput = 'genOutputHTML_' + str(validChunks) + '.html'
                tempReconstruct = str(validArray).replace(', ', '')
                dataReconstruct = tempReconstruct[1::]
                sumFilez = sum(map(int, fileZ))

                # Opens file
                file = open(dirPath + fileOutput, "w", 1048576)

                # HTML Header
                file.write("<html>")
                file.write("\n    <head>")
                file.write("\n        <title>Kubix Puzzle #{0:0=5d}</title>".format(int(validChunks)))
                file.write("\n        <meta />")
                file.write("\n        <link rel='stylesheet' type='text/css' href='style.css' />")
                file.write("\n    </head>")
                file.write("\n<body>")
                file.write("\n<div class='key'>")
                file.write("\n  <big>Kubix Puzzle</big>")
                file.write("\n  <br /><small>" + fileX + " " + fileY + " " + fileZ + "</small>")
                file.write("\n  <br />#{0:0=5d}".format(int(validChunks)))
                file.write("\n</div>")
                file.write("\n<div class='container'>")

                # Body Content
                for sections in range(dimCount):
                    file.write("\n    <div>")

                    # Sets dimensions (xyz)
                    for dimensions in range(layerCount):
                        outString = "\n       <table class='cube-" + str(dimensions) + "'>"
                        file.write(outString)

                        # Sets grids columns
                        for gridColumns in range(cube):
                            if gridColumns >= 1 and gridColumns < cube:
                                file.write("\n          </tr>")
                            file.write("\n          <tr>")

                            # Sets grids rows & content
                            for gridrows in range(cube):

                                if sections == 0:
                                    if dimensions == 0:
                                        popData = gridX[gridCount]
                                        getLocation = gridrows + (gridColumns * cube)
                                    elif dimensions == 1:
                                        popData = gridY[gridCount]
                                        getLocation = (gridrows + dimensions * cube) + (
                                        gridrows * ((cube ** (cubeDims-1)) - 1)) + (gridColumns * cube) - 1
                                    elif dimensions == 2:
                                        popData = gridZ[
                                            (cube * (cube - (gridColumns + 1)) + countZ) - (cube * gridColumns)]
                                        getLocation = ((cube ** cubeDims) - (cube ** (cubeDims-1) - (sections + 0 * cube)) - (
                                        (gridColumns * cube ** (cubeDims-1)))) + gridrows
                                        countZ += 1
                                    elif dimensions >= 3 and dimensions <= 6:
                                        getLocation = (((cube ** cubeDims) - (
                                        cube ** (cubeDims-1) - (sections + (dimensions - cubeDims) * cube)) - (
                                                        (gridColumns * cube ** (cubeDims-1)))) + gridrows)
                                        popData = dataReconstruct[getLocation]
                                    else:
                                        # Default case populates a space in HTML
                                        popData = '&nbsp;'

                                gridCount += 1

                                if gridCount >= cube**(cubeDims-1):
                                    gridCount = 0

                                file.write("\n              <td>")
                                file.write(str(popData))
                                if sections == 0 and dimensions <= 6:
                                    file.write("<small>" + str(getLocation) + "</small>")
                                file.write("</td>")

                        # Closes content in tables
                        file.write("\n          </tr>")
                        file.write("\n       </table>")

                        # Creates seperation hr tag
                        if dimensions == 6:
                            file.write("\n       <hr />")

                    # Closes the 1/3rd div containers
                    file.write("\n    </div>")

                # Footer
                file.write("\n </div>")
                file.write("\n</body>")
                file.write("\n</html>")

                # Close file
                file.close()

                if genHTML_verbose::
                    """ Pause output, without this it will take a long time and a lot of space
                        to generate all outputs """

                    print("\nVisualization #{0:0=5d} Generated".format(int(validChunks)))
                    pause = raw_input("Press Enter to continue...")

            # '''
            # Debugging
            # print('filez', fileZ)
            # print('Output test: ', validArray[:])
            # print('Validation index: ', outList[:])
            # print('Output: [%s] [%s] %s' % (chunkCountX, fileZ, validArray[:]))
            # pause = raw_input("Press Enter to continue...")
            # '''

            # Unknown: Determining if the input is loosely packed, archive compressing it may reduce
            #   amount of likely collisions and/or alternatively 'correct' outputs

            # "Mirror/Fix" Logic?
            #   Mirrors two valid groups (in 3d) as needed (not sure this will be needed).

            # "Collision Check" using "Known Data"
            #   Diagonal Logic is the simpliest but posibly has the highest overhead, see collision notes

            # "File Reconstruction"
            #   Converts arrays into final/original encoding (using a different name/extension?) & builds output file

            # Archival Compression
            #   Archives in zlib or equiv final output into a kbx file, saving ~66% to 75% final compression

            # "Performance Logging"
            #   Generates performance log and verbose output to file for use in GUI File Manager

            # "Assembly Compiler" API
            #   Script that compiles final determined "performance" version into static assembly code and
            #   "assembles" to all hardware platforms (x86, arm, pocket pc, etc)

            # "Cleanup"
            #   Cleans up all temp folders used, memory, etc (might be done automatically)
            #   Cleanup and optimize code itself, logic, and reduce redundant code

            # Testing/Debugging
            #   Try re-encoding the binary values and compare with original output (just to for valid "chunks")

            # "Other"
            # - Build GUI tools, services, documentation, etc.
            # - Build API, Debugging and Advanced Testing Tools,
            # - Build in "debugging" logic, and pass-through arguments
            # - Reduce and/or eliminate all dependencies,

            # if haultKubix: and validChunks == haultValue:
            #    break

        elif validateSumZ == validateSumY and not validateSumX:
            print('X Dimension contains an error, Resend X')
        elif validateSumZ == validateSumX and not validateSumY:
            print('Y Dimension contains an error, resend Y')
        elif validateSumX == validateSumY and not validateSumZ:
            print('Z Dimension contains an error, resend Z')
        else:
            print('Likely multiple transmission errors, resend chunk!\n')
            if multipleErrorCheck == 5:
                print('Data corrupt. Unable to recover current data chunk.')
                # Continue on error? (yes/no)
                # input('Ignore this "chunk" of data and continue? (Y/N)')
            else:
                multipleErrorCheck += 1
                # Repeat and re-check

print("Valid Chunks, chunkCount, validChunks", chunkCountX, chunkCountY, validChunks)
print("Unit test: Total [%s] valid [%s] invalid %s" % (sum(map(int, iArray)), iArray[0], iArray[1:]))
if enableLogTime:
    print("Total Run Time: %f seconds" % (time.time() - start_time))
