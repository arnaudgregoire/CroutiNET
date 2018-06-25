from PIL import Image
import os
import csv

"""
File used to clean row in dictonaries that are not linked with any pictures
"""
base_dir = r"D:\Arnaud\data_croutinet\paris"
roads_dictionnary_dir = os.path.join(base_dir,"roads_dictionnary.csv")
roads_correct_dictionnary_dir = os.path.join(base_dir,"roads_correct_dictionnary.csv")

img_dir = os.path.join(base_dir,"roads")

def checkImage(name):
    """
    check if the given name is really corresponding to a pictures in pictures folder
    :param name: the name you want to ckheck
    :return: boolean True or false depending if the file exist in pictures folder or not
    """
    print(name)
    path = os.path.join(img_dir, name)
    return os.path.isfile(path)


correctLines = []

with open(roads_dictionnary_dir, 'r') as csvfileReader:
    reader = csv.reader(csvfileReader, delimiter=',')

    for row in reader:
        line = row

        if(line != []):
            if (checkImage("2017_" + line[4] + "_" + line[3] + '.jpg' )):
                correctLines.append(line)



with open(roads_correct_dictionnary_dir, 'w') as csvfileWriter:
    writer = csv.writer(csvfileWriter, delimiter=',')
    for i in range(len(correctLines)):
        writer.writerow(correctLines[i])


