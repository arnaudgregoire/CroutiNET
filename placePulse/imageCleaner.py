from PIL import Image
import os
import csv

"""
File used to clean wrong or error pictures in a given folder
"No images found" pictures have a special size and thats how we detect them
"""
baseDir = "C:/Users/msawada/Desktop/arnaud/croutinet/placePulse/data"

wealthyDir = os.path.join(baseDir,"votesWealthy.csv")
correctWealthyDir = os.path.join(baseDir,"votesCorrectWealthy.csv")

imgDir = os.path.join(baseDir,"img")
correctImgDir = os.path.join(baseDir,"correctImg")

ERROR_SIZE = 2975
NO_IMAGE_SIZE = 3222


def checkImage(name):
    path = os.path.join(imgDir, name)
    correct = True

    if os.path.isfile(path):
        #print(os.stat(path).st_size)
        if(os.stat(path).st_size == ERROR_SIZE or os.stat(path).st_size == NO_IMAGE_SIZE):
            correct = False

    else:
        correct = False

    return correct

def copyImage(name):

    path = os.path.join(imgDir, name)
    im = Image.open(path)
    im.save(os.path.join(correctImgDir, name))


correctLines = []

with open(wealthyDir, 'r') as csvfileReader:
    reader = csv.reader(csvfileReader, delimiter=',')

    for row in reader:
        line = row

        if(line != []):
            if (checkImage(line[0] + '.jpg' ) and checkImage(line[1] + '.jpg')):
                correctLines.append(line)



with open(correctWealthyDir, 'w') as csvfileWriter:
    writer = csv.writer(csvfileWriter, delimiter=',')
    for i in range(len(correctLines)):
        writer.writerow(correctLines[i])


#for i in range(len(correctLines)):
#    copyImage(correctLines[i][0] + '.jpg')
#    copyImage(correctLines[i][1] + '.jpg')