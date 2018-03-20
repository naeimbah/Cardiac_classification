# import libraries
import numpy as np

from math import floor
from random import shuffle

def get_randomized_listed_tuples(data, curr_keys):
    """"
    INPUTS:
        data:
            the h5py data object
        curr_values:
            list of the current study-series keys to use
    OUTPUT:
        using a list of keys, will return a list of shuffled tuples
            [0]: the list of input keys
            [1]: the list of output keys
    """

    # initialize result list
    rslt_lst = []

    for curr_k in curr_keys:
        # get input list
        input_lst = list(data[curr_k + "/input"].keys())
        input_lst = ["{}/{}".format(curr_k + "/input", x) for x in input_lst]

        # make output list
        output_lst = [curr_k + "/output"] * len(input_lst)

        # add to return list
        [rslt_lst.append(x) for x in list(zip(input_lst, output_lst))]

    shuffle(rslt_lst)

    return rslt_lst

def get_batched_hdf5(data, tuple_lst):
    """"
    INPUTS:
        data:
            the h5py data object
        tuple_lst:
            [0]: the list of input keys
            [1]: the list of output keys (no need for specifying "/output")
    OUTPUT:
        [0]: stacked input_mtx
        [1]: stacked output_mtx
    """
    # get tuple list values
    input_keys = [x[0] for x in tuple_lst]
    output_keys = [x[1] for x in tuple_lst]

    # return lists of input and output
    input_lst = [data[x][()] for x in input_keys]
    output_lst = [data[x][()] for x in output_keys]

    # reshape input matrix
    input_lst = [np.expand_dims(x.astype('float32'), axis=-1) for x in input_lst]

    # stack
    input_rtn = np.stack(input_lst, axis=0)
    output_rtn = np.stack(output_lst, axis=0)

    return input_rtn, output_rtn

def data_generator(data, tuple_lst, batch_size):
    """"
    INPUTS:
        data:
            the h5py data object
        tuple_lst:
            [0]: the list of input keys
            [1]: the list of output keys (no need for specifying "/output")
        batch_size:
            the number of datasets used
   OUTPUT:
        [0]: input into network
        [1]: output into network
    """
    # make batch sized indicies
    min_cuts = floor(len(tuple_lst)/ batch_size)
    slices = np.arange(0, min_cuts*batch_size).reshape(min_cuts, batch_size).tolist()

    # if there's a remainder, append at end of list
    if len(tuple_lst) % batch_size:
        slices.append(np.arange(min_cuts*batch_size, len(tuple_lst)).tolist())

    # loop through values
    while 1:
        for curr_idx in slices:
            yield get_batched_hdf5(data, [tuple_lst[x] for x in curr_idx])
