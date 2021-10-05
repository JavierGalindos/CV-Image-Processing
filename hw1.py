# Javier Galindos Vicente
# Computer Vision 2021
"""
ITS8030: Homework 1

Please implement all functions below.

For submission create a project called its8030-2021-hw1 and put the solution in there.

Please note that NumPy arrays and PyTorch tensors share memory represantation, so when converting a
torch.Tensor type to numpy.ndarray, the underlying memory representation is not changed.

There is currently no existing way to support both at the same time. There is an open issue on
PyTorch project on the matter: https://github.com/pytorch/pytorch/issues/22402

There is also a deeper problem in Python with types. The type system is patchy and generics
has not been solved properly. The efforts to support some kind of generics for Numpy are
reflected here: https://github.com/numpy/numpy/issues/7370 and here: https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY
but there is currently no working solution. For Dicts and Lists there is support for generics in the 
typing module, but not for NumPy arrays.
"""
import cv2
import numpy as np
import util
import math

"""
Task 1: Convolution

Implement the function 

convolution(image : np.ndarray, kernel : np.ndarray, kernel_width : int, kernel_height : int, add : bool, in_place:bool) -> np.ndarray

to convolve an image with a kernel of size kernel_height*kernel_width.
Use zero-padding around the borders for simplicity (what other options would there be?).
Here:

    image is a 2D matrix of class double
    kernel is a 2D matrix with dimensions kernel_width and kernel_height
    kernel_width and kernel_height are the width and height of the kernel respectively

(Note: in the general case, they are not equal and may not be always odd, so you have to ensure that they are odd.)

    if add is true, then 128 is added to each pixel for the result to get rid of negatives.
    if in_place is False, then the output image should be a copy of the input image. The default is False,
    i.e. the operations are performed on the input image.

Write a general convolution function that can handle all possible cases as mentioned above.
You can get help from the convolution part of the function mean_blur_image (to be implemented in a lab)
to write this function.
"""
# Ref[1] : https://github.com/detkov/Convolution-From-Scratch

def add_padding(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Adds padding to the matrix.
    Args:
        image (np.ndarray): Matrix that needs to be padded. 2D matrix of class double
        kernel (np.ndarray): 2D matrix with dimensions kernel_width and kernel_height
    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = image.shape
    r = (kernel.shape[0] - 1) // 2
    c = (kernel.shape[1] - 1) // 2

    padded_image = np.zeros((n + r * 2, m + c * 2))
    padded_image[r: n + r, c: m + c] = image

    return padded_image


def convolution(image: np.ndarray, kernel: np.ndarray, kernel_width: int,
                kernel_height: int, add: bool, in_place: bool = False) -> np.ndarray:

    if not in_place:
        image = image.copy()

    h, w = image.shape

    # Check parameters
    kernel_is_correct = kernel_width % 2 == 1 and kernel_height % 2 == 1
    assert kernel_is_correct, 'Kernel shape should be odd.'
    matrix_to_kernel_is_correct = h >= kernel_height and w >= kernel_width
    assert matrix_to_kernel_is_correct, 'Kernel can\'t be bigger than matrix in terms of shape.'

    # Add zero padding to the input image
    r = (kernel.shape[0] - 1) // 2  # calculate padding
    c = (kernel.shape[1] - 1) // 2  # calculate padding
    padded_image = np.zeros((h + r * 2, w + c * 2))
    padded_image[r: h + r, c: w + c] = image

    # Convolution
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    b = r, c
    center_x_0 = b[0]
    center_y_0 = b[1]
    matrix_out = np.zeros((h, w))
    for i in range(h):
        center_x = center_x_0 + i
        indices_x = [center_x + l for l in range(-b[0], b[0] + 1)]
        for j in range(w):
            center_y = center_y_0 + j
            indices_y = [center_y + l for l in range(-b[1], b[1] + 1)]

            submatrix = padded_image[indices_x, :][:, indices_y]

            matrix_out[i][j] = np.sum(np.multiply(submatrix, kernel)) + add * 128

    matrix_out = np.clip(matrix_out, 0., 255., matrix_out)
    return matrix_out


"""
Task 2: Gaussian blur

Implement the function

gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray 

to Gaussian blur an image. "sigma" is the standard deviation of the Gaussian.
Use the function mean_blur_image as a template, create a 2D Gaussian filter
as the kernel and call the convolution function of Task 1.
Normalize the created kernel using the function normalize_kernel() (to
be implemented in a lab) before convolution. For the Gaussian kernel, use
kernel size = 2*radius + 1 (same as the Mean filter) and radius = int(math.ceil(3 * sigma))
and the proper normalizing constant.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0,
and save as "task2.png".
"""


def gaussian_blur_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    if not in_place:
        image = image.copy()

    # Get Gaussian Kernel
    radius = int(math.ceil(3 * sigma))
    ker_lenght = 2*radius + 1
    kernel = util.gkern(ker_lenght, sigma)

    # Normalize kernel
    kernel_norm = util.normalize_kernel(kernel)

    # Convolution
    blurred_img = convolution(image, kernel_norm, ker_lenght, ker_lenght, add=False, in_place=False)

    return blurred_img



"""
Task 3: Separable Gaussian blur

Implement the function

separable_gaussian_blur_image (image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray

to Gaussian blur an image using separate filters. "sigma" is the standard deviation of the Gaussian.
The separable filter should first Gaussian blur the image horizontally, followed by blurring the
image vertically. Call the convolution function twice, first with the horizontal kernel and then with
the vertical kernel. Use the proper normalizing constant while creating the kernel(s) and then
normalize using the given normalize_kernel() function before convolution. The final image should be
identical to that of gaussian_blur_image.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0, and save as "task3.png".
"""


def separable_gaussian_blur_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    if not in_place:
        image = image.copy()

    # Get Gaussian Kernel
    radius = int(math.ceil(3 * sigma))
    ker_lenght = 2*radius + 1

    kernel_horz = util.gkern1D(ker_lenght, sigma)
    kernel_vert = util.gkern1D(ker_lenght, sigma)

    # Normalize kernels
    kernel_norm_horz = util.normalize_kernel(kernel_horz)
    kernel_norm_vert = util.normalize_kernel(kernel_vert)

    # Convolution
    blurred_img_horz = convolution(image, kernel_norm_horz, ker_lenght, 1, add=False, in_place=False)
    blurred_img = convolution(blurred_img_horz, kernel_norm_horz, 1, ker_lenght, add=False, in_place=False)

    return blurred_img


"""
Task 4: Image derivatives

Implement the functions

first_deriv_image_x(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray
first_deriv_image_y(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray and
second_deriv_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray

to find the first and second derivatives of an image and then Gaussian blur the derivative
image by calling the gaussian_blur_image function. "sigma" is the standard deviation of the
Gaussian used for blurring. To compute the first derivatives, first compute the x-derivative
of the image (using the horizontal 1*3 kernel: [-1, 0, 1]) followed by Gaussian blurring the
resultant image. Then compute the y-derivative of the original image (using the vertical 3*1
kernel: [-1, 0, 1]) followed by Gaussian blurring the resultant image.
The second derivative should be computed by convolving the original image with the
2-D Laplacian of Gaussian (LoG) kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] and then applying
Gaussian Blur. Note that the kernel values sum to 0 in these cases, so you don't need to
normalize the kernels. Remember to add 128 to the final pixel values in all 3 cases, so you
can see the negative values. Note that the resultant images of the two first derivatives
will be shifted a bit because of the uneven size of the kernels.

To do: Compute the x-derivative, the y-derivative and the second derivative of the image
"cactus.jpg" with a sigma of 1.0 and save the final images as "task4a.png", "task4b.png"
and "task4c.png" respectively.
"""


def first_deriv_image_x(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    if not in_place:
        image = image.copy()

    # Create kernel for 1st derivative
    kernel_diffx = np.array([[-1, 0, 1]])

    # Get Gaussian Kernel
    radius = int(math.ceil(3 * sigma))
    ker_lenght = 2 * radius + 1
    kernel = util.gkern(ker_lenght, sigma)

    # Normalize kernel
    kernel_norm = util.normalize_kernel(kernel)

    # Convolution
    diff_im = convolution(image, kernel_diffx, kernel_diffx.shape[0], kernel_diffx.shape[0], add=False, in_place=False)
    blurred_img = convolution(diff_im, kernel_norm, ker_lenght, ker_lenght, add=False, in_place=False)

    # Avoid negative values
    img_non_negative = np.array(blurred_img + 128)
    img_non_negative = np.clip(img_non_negative, 0., 255., img_non_negative)

    return img_non_negative


def first_deriv_image_y(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    if not in_place:
        image = image.copy()

    # Create kernel for 1st derivative
    kernel_diffy = np.array([[-1], [0], [1]])

    # Get Gaussian Kernel
    radius = int(math.ceil(3 * sigma))
    ker_lenght = 2 * radius + 1
    kernel = util.gkern(ker_lenght, sigma)

    # Normalize kernel
    kernel_norm = util.normalize_kernel(kernel)

    # Convolution
    diff_im = convolution(image, kernel_diffy, kernel_diffy.shape[0], kernel_diffy.shape[0], add=False, in_place=False)
    blurred_img = convolution(diff_im, kernel_norm, ker_lenght, ker_lenght, add=False, in_place=False)

    # Avoid negative values
    img_non_negative = np.array(blurred_img + 128)
    img_non_negative = np.clip(img_non_negative, 0., 255., img_non_negative)

    return img_non_negative


def second_deriv_image(image: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    if not in_place:
        image = image.copy()

    # 2D Laplacian of Gaussian (LoG)
    kernel_log = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Get Gaussian Kernel
    radius = int(math.ceil(3 * sigma))
    ker_lenght = 2 * radius + 1
    kernel = util.gkern(ker_lenght, sigma)

    # Normalize kernel
    kernel_norm = util.normalize_kernel(kernel)

    # Convolution
    diff_im = convolution(image, kernel_log, kernel_log.shape[0], kernel_log.shape[0], add=False, in_place=False)
    blurred_img = convolution(diff_im, kernel_norm, ker_lenght, ker_lenght, add=False, in_place=False)

    # Avoid negative values
    img_non_negative = np.array(blurred_img + 128)
    img_non_negative = np.clip(img_non_negative, 0., 255., img_non_negative)

    return img_non_negative


"""
Task 5: Image sharpening

Implement the function
sharpen_image(image : np.ndarray, sigma : float, alpha : float, in_place : bool = False) -> np.ndarray
to sharpen an image by subtracting the Gaussian-smoothed second derivative of an image, multiplied
by the constant "alpha", from the original image. "sigma" is the Gaussian standard deviation. Use
the second_deriv_image implementation and subtract back off the 128 that second derivative added on.

To do: Sharpen "yosemite.png" with a sigma of 1.0 and alpha of 5.0 and save as "task5.png".
"""


def sharpen_image(image: np.ndarray, sigma: float, alpha: float, in_place: bool = False) -> np.ndarray:
    if not in_place:
        image = image.copy()

    second_diff = second_deriv_image(image, sigma)
    second_diff = np.array(second_diff - 128)

    sharp_img = image + alpha * second_diff

    return sharp_img


"""
Task 6: Edge Detection

Implement 
sobel_image(image : np.ndarray, in_place : bool = False) -> np.ndarray
to compute edge magnitude and orientation information. Convert the image into grayscale.
Use the standard Sobel masks in X and Y directions:
[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] and [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] respectively to compute
the edges. Note that the kernel values sum to 0 in these cases, so you don't need to normalize the
kernels before convolving. Divide the image gradient values by 8 before computing the magnitude and
orientation in order to avoid spurious edges. sobel_image should then display both the magnitude and
orientation of the edges in the image.

To do: Compute Sobel edge magnitude and orientation on "cactus.jpg" and save as "task6.png".
"""


def sobel_image(image: np.ndarray, in_place: bool = False) -> np.ndarray:
    "implement the function here"
    raise "not implemented yet!"


"""
Task 7: Bilinear Interpolation

Implement the function
bilinear_interpolation(image : np.ndarray, x : float, y : float) -> np.ndarray

to compute the linearly interpolated pixel value at the point (x,y) using bilinear interpolation.
Both x and y are real values. Put the red, green, and blue interpolated results in the vector "rgb".

To do: The function rotate_image will be implemented in a lab and it uses bilinear_interpolation
to rotate an image. Rotate the image "yosemite.png" by 20 degrees and save as "task7.png".
"""


def bilinear_interpolation(image: np.ndarray, x: float, y: float) -> np.ndarray:
    "Returns a  vector containing interpolated red green and blue values (a vector of length 3)"
    "implement the function here"
    raise "not implemented yet!"


def rotate_image(image: np.ndarray, rotation_angle: float, in_place: bool = False) -> np.ndarray:
    rgb = bilinear_interpolation(image, 0.5, 0.0)
    "To be implemented by the lecturer"
    raise "not implemented yet"


"""
Task 8: Finding edge peaks

Implement the function
find_peaks_image(image : np.ndarray, thres : float, in_place : bool = False) -> np.ndarray
to find the peaks of edge responses perpendicular to the edges. The edge magnitude and orientation
at each pixel are to be computed using the Sobel operators. The original image is again converted
into grayscale in the starter code. A peak response is found by comparing a pixel's edge magnitude
to that of the two samples perpendicular to the edge at a distance of one pixel, which requires the
bilinear_interpolation function
(Hint: You need to create an image of magnitude values at each pixel to send as input to the
interpolation function).
If the pixel's edge magnitude is e and those of the other two are e1 and e2, e must be larger than
"thres" (threshold) and also larger than or equal to e1 and e2 for the pixel to be a peak response.
Assign the peak responses a value of 255 and everything else 0. Compute e1 and e2 as follows:

(please check the separate task8.pdf)

To do: Find the peak responses in "virgintrains.jpg" with thres = 40.0 and save as "task8.png".
What would be a better value for thres?
"""


def find_peaks_image(image: np.ndarray, thres: float, in_place: bool = False) -> np.ndarray:
    "implement the function here"
    raise "not implemented yet!"


"""
Task 9 (a): K-means color clustering with random seeds (extra task)

Implement the function

random_seed_image(image : np.ndarray, num_clusters : int, in_place : bool = False) -> np.ndarray

to perform K-Means Clustering on a color image with randomly selected initial cluster centers
in the RGB color space. "num_clusters" is the number of clusters into which the pixel values
in the image are to be clustered. Use random.randint(0,255) to initialize each R, G and B value.
to create #num_clusters centers, assign each pixel of the image to its closest cluster center
and then update the cluster centers with the average of the RGB values of the pixels belonging
to that cluster until convergence. Use max iteration # = 100 and L1 distance between pixels,
i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30 (or anything suitable). Note: Your code
should account for the case when a cluster contains 0 pixels during an iteration. Also, since
this algorithm is random, you will get different resultant images every time you call the function.

To do: Perform random seeds clustering on "flowers.png" with num_clusters = 4 and save as "task9a.png".
"""


def random_seed_image(image: np.ndarray, num_clusters: int, in_place: bool = False) -> np.ndarray:
    "implement the function here"
    raise "not implemented yet!"


"""
Task 9 (b): K-means color clustering with pixel seeds (extra)

Implement the function
pixel_seed_image(image : np.ndarray, num_clusters: int, in_place : bool = False)
to perform K-Means Clustering on a color image with initial cluster centers sampled from the
image itself in the RGB color space. "num_clusters" is the number of clusters into which the
pixel values in the image are to be clustered. Choose a pixel and make its RGB values a seed
if it is sufficiently different (dist(L1) >= 100) from already-selected seeds. Repeat till
you get #num_clusters different seeds. Use max iteration # = 100 and L1 distance between pixels,
 i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
 the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30.

To do: Perform pixel seeds clustering on "flowers.png" with num_clusters = 5 and save as "task9b.png".
"""


def pixel_seed_image(image: np.ndarray, num_clusters: int, in_place: bool = False) -> np.ndarray:
    "implement the function here"
    raise "not implemented yet!"
