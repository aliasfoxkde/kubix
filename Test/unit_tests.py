def validate():
    global unitTest_Validate
    global unitTest_Verbose

    unitTest_Validate = True
    unitTest_Verbose = True


    # Unit test used for validation/verification of cube/chunk
    if unitTest_Validate == True:
        compareLocations, compareLocCheck, invalidLocations = list(), list(), list()
        currentZ, numOfValid, numOfInvalid = 0, 0, 0
        unitStatus = 'Successful'

        for b in range(0, cube):
            for a in range(0, cube):
                location = a + (b * (cube ** 2))

                for getSumLocation in range(cube):
                    compareLocations.append(validArray[a + (getSumLocation * cube) + (b * (cube * cube))])
                    compareLocCheck.append(a + (getSumLocation * cube) + (b * (cube ** 2)))

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

        if unitTest_Verbose == True:
            output = ''.join(map(str, validArray))

            print('\nUnit Test Summary:')
            print('  Chunk [%s] Length [%s]' % (validChunks, len(output)))
            print('  Array [%s]' % (output))
            print('  Validation check %s!' % unitStatus)
            print('  {} of {} Columns Valid ({} Invalid {})'.format(numOfValid, cube ** 2, numOfInvalid,
                                                                    invalidLocations))

            pause = raw_input("  Press Enter to continue...")

            # BUG: The Invalid locations appear to be off by 1 (need to shift up 1)?