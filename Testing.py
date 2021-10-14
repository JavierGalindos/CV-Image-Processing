import hw1
import cv2
import numpy as np
import helpers
from skimage.exposure import rescale_intensity

''' Task 1: Convolution'''
img = cv2.imread('images/flowers.png', cv2.IMREAD_GRAYSCALE)

cv2.startWindowThread()
cv2.imshow('Image', img)
cv2.waitKey(0)

k_size1 = 5
k_size2 = 3
sigma = 2

kernel = helpers.matlab_style_gauss2D((k_size1, k_size2), sigma)
convolved_img = hw1.convolution(img, kernel, k_size1, k_size2, add=False, in_place=False)

cv2.imshow('Image Convolved', convolved_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Test.jpg', convolved_img)

''' Convolution 3D'''
img = cv2.imread('images/flowers.png', cv2.IMREAD_COLOR)

cv2.startWindowThread()

cv2.imshow('Image', img)
cv2.waitKey(0)

k_size1 = 9
k_size2 = 9
sigma = 3

kernel = helpers.matlab_style_gauss2D((k_size1, k_size2), sigma)
convolved_img = hw1.convolution(img, kernel, k_size1, k_size2, add=False, in_place=False)

cv2.imshow('Image Convolved', convolved_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('Test.jpg', convolved_img)

''' Task 2: Gaussian Blur'''
img = cv2.imread('images/songfestival.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
cv2.waitKey(0)

sigma = 4.0
blurred_img = hw1.gaussian_blur_image(img, sigma)
blurred_img = blurred_img.astype("uint8")

cv2.imwrite('imagesOutput/task2.png', blurred_img)
cv2.imshow('Image', blurred_img)
cv2.waitKey(0)

''' Task 3: Separable Gaussian Blur'''

img = cv2.imread('images/songfestival.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
cv2.waitKey(0)

sigma = 4.0
blurred_img = hw1.separable_gaussian_blur_image(img, sigma)
blurred_img = blurred_img.astype("uint8")

cv2.imwrite('imagesOutput/task3.png', blurred_img)
cv2.imshow('Image', blurred_img)
cv2.waitKey(0)

''' Task 4:Diff Images'''

img = cv2.imread('images/cactus.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
cv2.waitKey(0)

sigma = 1.0
# Diff x
diffx = hw1.first_deriv_image_x(img, sigma)
diffx = diffx.astype("uint8")

cv2.imwrite('imagesOutput/task4a.png', diffx)
cv2.imshow('Image', diffx)
cv2.waitKey(0)

# Diff y
diffy = hw1.first_deriv_image_y(img, sigma)
diffy = diffy.astype("uint8")

cv2.imwrite('imagesOutput/task4b.png', diffy)
cv2.imshow('Image', diffy)
cv2.waitKey(0)

# Diff 2
diff2 = hw1.second_deriv_image(img, sigma)
diff2 = diff2.astype("uint8")

cv2.imwrite('imagesOutput/task4c.png', diff2)
cv2.imshow('Image', diff2)
cv2.waitKey(0)

''' Task 5: Sharpening'''

img = cv2.imread('images/yosemite.png', cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
cv2.waitKey(0)

sigma = 1.0
alpha = 5.0
sharpen_img = hw1.sharpen_image(img, sigma, alpha)

sharpen_img = hw1.rescale_int(sharpen_img)  # Rescale

cv2.imwrite('imagesOutput/task5.png', sharpen_img)
cv2.imshow('Image', sharpen_img)
cv2.waitKey(0)

''' Task 6: Edge detection'''

img = cv2.imread('images/cactus.jpg', cv2.IMREAD_COLOR)

magnitude, orientation = hw1.sobel_image(img)

magnitude = magnitude.astype("uint8")
orientation = rescale_intensity(orientation, in_range=(0., 1.)) # Change radian to pixels for visualization
orientation = orientation*255.
orientation = orientation.astype("uint8")

cv2.imwrite("imagesOutput/task6_magnitude.png", magnitude)
cv2.imshow('Image', magnitude)
cv2.waitKey(0)

cv2.imwrite("imagesOutput/task6_orientation.png", orientation)
cv2.imshow('Image', orientation)
cv2.waitKey(0)

# Other visualization for orientation
tmp = (magnitude > (np.max(magnitude)/8.)).astype(np.float32)
GA_scaled = (orientation - np.min(orientation))/(np.max(orientation) - np.min(orientation))
# Color of the angle of the gradient
GA_cm = np.zeros((orientation.shape[0], orientation.shape[1], 3), dtype=np.float32)
GA_cm[:,:,0] = GA_scaled*tmp
GA_cm[:,:,2] = GA_scaled*tmp
GA_cm[:,:,2] = (1- GA_cm[:,:,2])*tmp


cv2.imwrite("imagesOutput/task6_orientation&magnitude.png", (GA_cm * 255.).astype("uint8"))
cv2.imshow('Image', GA_cm)
cv2.waitKey(0)


''' Task 7: Rotating image'''
img = cv2.imread('images/yosemite.png', cv2.IMREAD_COLOR)

angle = 20.0
rot_img = hw1.rotate_image(img, angle)

cv2.imwrite("imagesOutput/task7.png", rot_img)
cv2.imshow('Image', rot_img)
cv2.waitKey(0)

''' Task 8: Find edge peaks'''
img = cv2.imread('images/virgintrains.jpg', cv2.IMREAD_COLOR)

thres = 25.
edge_peaks = hw1.find_peaks_image(img, thres)

cv2.imwrite("imagesOutput/task8.png", edge_peaks)
cv2.imshow('Image', edge_peaks)
cv2.waitKey(0)

''' Task 9: k-means random seeds '''
img = cv2.imread('images/flowers.png', cv2.IMREAD_COLOR)

num_cluster = 4

image_k = hw1.random_seed_image(img, num_cluster)

cv2.imwrite("imagesOutput/task9a.png", image_k)
cv2.imshow('Image', image_k)
cv2.waitKey(0)

image_k = hw1.pixel_seed_image(img, num_cluster)
cv2.imwrite("imagesOutput/task9b.png", image_k)
cv2.imshow('Image', image_k)
cv2.waitKey(0)


''' Destroy all windows '''
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
