import numpy as np
# import cv2
import matplotlib.pyplot as plt

def plotValues(t):
    plt.figure()
    plt.plot(np.arange(t.shape[0])+1, t, np.arange(t.shape[0])+1, t, 'ro')
    plt.ylabel("Temperature (°C)")
    plt.xlabel("Measurements")
    plt.show()

def calculatePadding(kernel_size): # DO NOT USE EVEN NUMBERS
    return (kernel_size - 1) // 2

def plotImage(image, cmap=None):
    plt.figure()
    if cmap != None:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp( -(x * x + y * y) / (2. * sigma * sigma) )
    h[ h < np.finfo(h.dtype).eps * h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h