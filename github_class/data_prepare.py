import numpy as np



def data_preprocess(X,test_x):
    X_normal = X
    test_x_normal = test_x
    for i in range(len(X)):
        X_normal[i,:,:,0] = image_histogram_equalization(X[i,:,:,0])[0]
    print('Train_set is normalized')
    for i in range(len(test_x)):
       test_x_normal[i,:,:,0] = image_histogram_equalization(test_x[i,:,:,0])[0]
    print('Test_set is normalized')

    return X_normal , test_x_normal

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf
