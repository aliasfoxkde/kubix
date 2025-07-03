            # Generates "Game" Outputs to HTML

            # Sets Paths
            dirPath = '/home/mkinney/PycharmProjects/kubix/htmlGameOutputs/'
            # fileOutput = 'genOutput1.html'
            fileOutput = 'genOutputHTML_' + str(validChunks) + '.html'

            # Data values
            tempReconstruct = str(outputArrayX).replace(', ','')
            dataReconstruct = tempReconstruct[1::]
            gridX = str(filex)
            gridY = str(filey)
            gridZ = str(filez)

            gridValid = 'No'
            gridChk = 33
            gridCount = 0
            popData = 0
            countZ = 0

            # Sets variables
            dimCount = 3
            dimResetCount = 1
            layerCount = 11
            cube = 4

            # Opens file
            file = open(dirPath + fileOutput, "w", 1048576)

            # HTML Header
            file.write("<html>")
            file.write("\n    <head>")
            file.write("\n        <title></title>")
            file.write("\n        <meta />")
            file.write("\n        <link rel='stylesheet' type='text/css' href='style.css' />")
            file.write("\n    </head>")
            file.write("\n<body>")
            file.write("\n<div class='key'>")
            file.write("\n  <big>Kubix Puzzle</big>" )
            # file.write("\n  <br /><small>" + filex + " " + filey + " " + filez + "</small>")
            file.write("\n  <br />#{0:0=5d}".format(int(validChunks)))
            file.write("\n</div>")
            file.write("\n<div class='container'>")

            # Body Content
            for sections in xrange(dimCount):
                file.write("\n    <div>")

                # Sets dimensions (xyz)
                for dimensions in xrange(layerCount):
                    outString = "\n       <table class='cube-" + str(dimensions) + "'>"
                    file.write(outString)

                    # Sets grids columns
                    for gridColumns in xrange(cube):
                        if gridColumns >= 1 and gridColumns < cube:
                            file.write("\n          </tr>")

                        file.write("\n          <tr>")

                        # Sets grids rows & content
                        for gridrows in xrange(cube):

                            # print(getLocation)
                            # pause = raw_input("Press Enter to continue...")

                            if sections == 0:
                                if dimensions == 0:
                                    popData = gridX[gridCount]
                                    getLocation = gridrows+(int(gridColumns)*cube)
                                elif dimensions == 1:
                                    popData = gridY[gridCount]
                                    getLocation = (gridrows+dimensions*cube)+(gridrows*((cube**2)-1))+(gridColumns*cube)-1
                                elif dimensions == 2:
                                    popData = gridZ[(cube*(cube-(gridColumns+1))+countZ)-(cube*gridColumns)]
                                    getLocation = (((cube**3)-(cube*cube-(sections+0*cube))-((gridColumns*cube**2)))+(gridrows))
                                    countZ += 1
                                elif dimensions >= 3 and dimensions <= 6:
                                    getLocation = (((cube**3)-(cube*cube-(sections+(dimensions-3)*cube))-((gridColumns*cube**2)))+(gridrows))
                                    popData = dataReconstruct[getLocation]
                                else:
                                    popData = '&nbsp;'

                            gridCount += 1

                            if gridCount >= 16:
                                gridCount = 0

                            file.write("\n              <td>")
                            file.write(str(popData))
                            if sections == 0 and dimensions <= 6:
                                file.write("<small>" + str(getLocation)+ "</small>")
                            file.write("</td>")

                            '''
                            # Debugging
                            print(gridCount)
                            pause = raw_input("Press Enter to continue...")
                            '''

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

            # outTest = str(outputArrayX).replace(', ','')
            # print(outTest)
            # print(outTest[1::])
            pause = raw_input("Press Enter to continue...")

            # Resets variables and lists
            xCount = 0 # Row
            yCount = 0 # Column
            zCount = 0 # Layer
            validArray = outputArrayX # Temporary way to contain array for testing only
            compareLocations = []
            debugValues = []
