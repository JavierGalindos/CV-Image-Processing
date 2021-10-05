import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/javiergalindos/OneDrive - Universidad Politécnica de Madrid/Documentos/MSc DM/2ºDM/Computer Vision/Code/hw1')


import hw1
import cv2
import numpy as np
import helpers
from matplotlib import pyplot as plt

''' Convolution'''

img = cv2.imread('hw1/images/flowers.png',cv2.IMREAD_GRAYSCALE)


cv2.imshow('Image',img)
cv2.waitKey(0)

k_size1 = 5
k_size2 = 3
sigma = 2


kernel = helpers.matlab_style_gauss2D((k_size1,k_size2), sigma)
convolved_img = hw1.convolution(img,kernel,k_size1,k_size2, add=False, in_place=False)

cv2.imshow('Image Convolved',convolved_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Test_gray.jpg', convolved_img)

''' Gaussian Blur'''

img = cv2.imread('hw1/images/songfestival.jpg' ,cv2.IMREAD_GRAYSCALE)
cv2.imshow('Image',img)
cv2.waitKey(0)

sigma = 4.0

blurred_img = hw1.gaussian_blur_image(img,sigma)

cv2.imwrite('task2.png', blurred_img)
cv2.imshow('Image',blurred_img)
cv2.waitKey(0)

plt.imshow(blurred_img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

''' Sperable Gaussian Blur'''

img = cv2.imread('hw1/images/songfestival.jpg' ,cv2.IMREAD_GRAYSCALE)
cv2.imshow('Image',img)
cv2.waitKey(0)

sigma = 4.0

blurred_img = hw1.separable_gaussian_blur_image(img,sigma)

cv2.imwrite('task3.png', blurred_img)
cv2.imshow('Image',blurred_img)
cv2.waitKey(0)

plt.imshow(blurred_img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

''' Diff Images'''

img = cv2.imread('hw1/images/cactus.jpg' ,cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

sigma = 1.0
# Diff x
blurred_img = hw1.first_deriv_image_x(img,sigma)

cv2.imwrite('task4a.png', blurred_img)
cv2.imshow('Image',blurred_img)
cv2.waitKey(0)

plt.imshow(blurred_img, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# Diff y
blurred_img = hw1.first_deriv_image_y(img,sigma)

cv2.imwrite('task4b.png', blurred_img)
cv2.imshow('Image',blurred_img)
cv2.waitKey(0)

plt.imshow(blurred_img, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# Diff 2
blurred_img = hw1.second_deriv_image(img,sigma)

cv2.imwrite('task4c.png', blurred_img)
cv2.imshow('Image',blurred_img)
cv2.waitKey(0)

plt.imshow(blurred_img, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

''' Sharpening'''

img = cv2.imread('hw1/images/yosemite.png' ,cv2.IMREAD_GRAYSCALE)
cv2.imshow('Image',img)
cv2.waitKey(0)

sigma = 1.0
alpha = 5.0

sharpen_img = hw1.sharpen_image(img,sigma,alpha)

cv2.imwrite('task5.png', sharpen_img)
cv2.imshow('Image',blurred_img)
cv2.waitKey(0)

plt.imshow(sharpen_img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()