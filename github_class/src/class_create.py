import dicom
import os
import numpy as np
import scipy.misc


# HDF5 materials
import numpy as np
import h5py
import skimage.transform

def class_dict_get(path0,path1,path2):
    class_dict = {}
    path_dict = {}
    #PathDicom = "/data/SAX-training-spiderman/SAX-training"

    for dirName, subdirList, fileList in os.walk(path0):
        for filename in fileList:

            if ".dcm" in filename.lower():  # check whether the file's DICOM

                drive, path_and_file = os.path.splitdrive(dirName)
                path, file = os.path.split(path_and_file)
                series = file
                path, file = os.path.split(path)
                study = file
                class_dict[study+'_'+series] = 0
                path_dict[dirName+'/'+filename] = 0


    #PathDicom = "/data/Auto-pilot training data"
    for dirName, subdirList, fileList in os.walk(path1):
        for filename in fileList:

            if ".dcm" in filename.lower():  # check whether the file's DICOM

               drive, path_and_file = os.path.splitdrive(dirName)
               path, file = os.path.split(path_and_file)
               series = file
               path, file = os.path.split(path)
               study = file
               class_dict[study+'_'+series] = 1
               path_dict[dirName+'/'+filename] = 1


   #PathDicom = "/data/VLAX-training"
    for dirName, subdirList, fileList in os.walk(path2):
        for filename in fileList:

            if ".dcm" in filename.lower():  # check whether the file's DICOM

               drive, path_and_file = os.path.splitdrive(dirName)
               path, file = os.path.split(path_and_file)
               series = file
               path, file = os.path.split(path)
               study = file
               class_dict[study+'_'+series] = 2
               path_dict[dirName+'/'+filename] = 2

    return class_dict,path_dict
