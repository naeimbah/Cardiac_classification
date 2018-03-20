#!/usr/bin/env python

# load libraries
import os
import re
import h5py
import dicom

import numpy as np

from functools import partial
from multiprocessing import Pool
from skimage.transform import resize

from src.interpolate import linear_interpolate
from src.class_create import class_dict_get

# script level variables
DICOM_PATTERN = re.compile(".dcm$")
STUDY_SERIES_ID_REGEX = re.compile("[0-9]+")
path0 = "/data/SAX-training-spiderman/SAX-training"
path1 = "/data/Auto-pilot"
path2 = "/data/VLAX-training"
[class_dict,path_dict] = class_dict_get(path0,path1,path2)

settings_dict = {
        "target_size": (128, 128),
        "sort_apical": False,
        "interpolate": None, # removed interpolation
        "data_path": "/data/ex_data.hdf5"
                 }


def process_hdf5(path0, path1, path2, class_dict, settings_dict):
    """
    INPUT:
        base_dir:
            the base directory containing studies
        settings_dict:
            [target_size]: the target Y Z dimentions
            [sort_apical]: if we sort apically
            [interpolate]: do we interpolate to a set number of images or None
            [data_path]: the path for the data path
        class_dict:
            [key]: the study_series name
            [values]: the class defination 'int'
    EFFECT:
        creates a HDF5 file in the base data dir
            *** WILL OVERWRITE OLD HDF5 FILE! ***
    """
    # verify settings_dict has correct keys
    settings_key_lst = ["target_size", "sort_apical", "interpolate", "data_path"]
    for key in settings_key_lst:
        if not key in settings_dict:
            raise KeyError("{} not in settings_dict".format(key))

    # list studies
    study_lst0 = os.listdir(path0)
    study_lst0 = [x for x in study_lst0 if STUDY_SERIES_ID_REGEX.search(x)]

    study_lst1 = os.listdir(path1)
    study_lst1 = [x for x in study_lst1 if STUDY_SERIES_ID_REGEX.search(x)]

    study_lst2 = os.listdir(path2)
    study_lst2 = [x for x in study_lst2 if STUDY_SERIES_ID_REGEX.search(x)]

    study_lst = study_lst0 + study_lst1 + study_lst2


    # list series
    series_lst0 = [os.listdir(os.path.join(path0, x)) for x in study_lst0]
    series_lst0 = [[y for y in x if STUDY_SERIES_ID_REGEX.search(y)] for x in series_lst0]

    series_lst1 = [os.listdir(os.path.join(path1, x)) for x in study_lst1]
    series_lst1 = [[y for y in x if STUDY_SERIES_ID_REGEX.search(y)] for x in series_lst1]

    series_lst2 = [os.listdir(os.path.join(path2, x)) for x in study_lst2]
    series_lst2 = [[y for y in x if STUDY_SERIES_ID_REGEX.search(y)] for x in series_lst2]

    series_lst = series_lst0 + series_lst1 + series_lst2

    # combine study and series
    path_lst0 = [[os.path.join(x[0], y) for y in x[1]] for x in zip(study_lst0, series_lst0)]
    path_lst0 = [x for y in path_lst0 for x in y]
    path_lst0 = [os.path.join(path0, x) for x in path_lst0]

    path_lst1 = [[os.path.join(x[0], y) for y in x[1]] for x in zip(study_lst1, series_lst1)]
    path_lst1 = [x for y in path_lst1 for x in y]
    path_lst1 = [os.path.join(path1, x) for x in path_lst1]

    path_lst2 = [[os.path.join(x[0], y) for y in x[1]] for x in zip(study_lst2, series_lst2)]
    path_lst2 = [x for y in path_lst2 for x in y]
    path_lst2 = [os.path.join(path2, x) for x in path_lst2]

    path_lst = path_lst0 + path_lst1 + path_lst2

    # get list of class labels
    key_lst = [[(x[0]+'_'+y) for y in x[1]] for x in zip(study_lst, series_lst)]
    key_lst = [x for y in key_lst for x in y]

    class_lst = [class_dict[x] for x in key_lst]

    # convert to tuple
    tuple_lst = list(zip(path_lst, class_lst, key_lst))

    # make new hdf5 file
    if os.path.isfile(settings_dict["data_path"]):
        print("Attempting to remove old file at: {}".format(settings_dict["data_path"]))
        try:
            os.remove(settings_dict["data_path"])
        except:
            print("Could not remove old file. Try closing old file. Exiting.")

    # make file link
    data_file = h5py.File(settings_dict["data_path"])

    # make partial
    func = partial(
        process_dicom_series, # func
        settings_dict["target_size"],
        settings_dict["sort_apical"],
        settings_dict["interpolate"],
    )

    # parrallel computing
    p = Pool()

    # map function
    print("\n\nProcessing data:\n")
    storage = p.map(func, tuple_lst)
    #storage = [process_dicom_series(settings_dict["target_size"], settings_dict["sort_apical"], settings_dict["interpolate"], x) for x in tuple_lst]

    print("\n")
    # remove None's
    storage = list(filter(lambda x: x is not None, storage))

    # combine dicts
    storage_dict = {k: v for x in storage for k, v in x.items()}

    # write data
    for k, v in storage_dict.items():
        data_file[k] = v

    # clean up
    p.close()
    p.join()
    data_file.close()

def process_dicom_series(target_size, sort_apical, interpolate, input_tuple):
    """
    INPUT:
        target_size:
            the tuple specifying resolution (Y, X)
        sort_apical:
            do we sort apical or not
        interpolate:
            if False, no interpolation
            if int, interpolate to that number of slices
        input_tuple:
            [0]:
                the input path specifying a series directory filled with dicom files
            [1]:
                the 'int' class label
            [2]:
                the study_series key
    OUTPUT:
        the dictionary containing the labeled dicom files with a label
    """
    # get tuple values
    input_path = input_tuple[0]
    class_label = input_tuple[1]
    study_series_id = input_tuple[2]

    # list only dicom files
    f_lst = os.listdir(input_path)
    try:
        f_lst = [x for x in f_lst if DICOM_PATTERN.search(x).group()]
    except AttributeError:
        f_lst = [x for x in f_lst if DICOM_PATTERN.search(x)]
    f_lst = [os.path.join(input_path, x) for x in f_lst]

    # read in dicom files
    dicom_lst = [dicom.read_file(x, force=True) for x in f_lst]
    dicom_lst = sorted(dicom_lst, key=lambda dicom: dicom.InstanceNumber)

    # sort apical
    if sort_apical:
        if dicom_lst[0].ImagePositionPatient[1] > dicom_lst[-1].ImagePositionPatient[1]:
            reversed(dicom_lst)

    # convert to image
    img_lst = [x.pixel_array for x in dicom_lst]

    # resize
    img_lst = [resize(x, target_size, mode="constant") for x in img_lst]

    # interpolate if neccessary
    if type(interpolate) == int:
        mtx = linear_interpolate(img_lst, interpolate)
        img_lst = [mtx[:, :, x] for x in range(interpolate)]

    # move to temp dictionary
    temp_storage = {}
    temp_storage[study_series_id + "/output"] = np.int(class_label)

    # add images
    for img_indx in range(len(img_lst)):
        curr_img = img_lst[img_indx].astype('float32')
        temp_storage[study_series_id + "/input/" + str(img_indx)] = curr_img

    return temp_storage
