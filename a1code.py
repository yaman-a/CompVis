
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import rotate

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    # YOUR CODE HERE

    img = io.imread(img_path)
    img_array = np.array(img)
    img_array = np.divide(img_array, 255)
    
    
    return img_array

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    
    # YOUR CODE HERE

    print(image.shape)

    
    return None

def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index 
        start_col (int): The starting column index 
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

     ### YOUR CODE HERE

    return image[start_row:start_row + num_rows, start_col:start_col + num_cols]

   


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, change 0.5 to 128.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    ### YOUR CODE HERE

    out = factor * (image - 0.5) + 0.5

    out = np.clip(out, 0, 1)

    return out


def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    ### commenting out because this one can't handle greyscale
    # # get input image dimensions
    # input_rows, input_cols, channels = input_image.shape  

    # # compute the scaling factors
    # row_scale = input_rows / output_rows
    # col_scale = input_cols / output_cols

    # # create an empty array for the resized image
    # out = np.zeros((output_rows, output_cols, channels))

    # # apply nearest neighbor interpolation
    # for i in range(output_rows):
    #     for j in range(output_cols):
    #         # Find the nearest neighbor in the original image
    #         orig_i = int(i * row_scale)
    #         orig_j = int(j * col_scale)

    #         # assign the nearest pixel value
    #         out[i, j] = input_image[orig_i, orig_j]
    
    # return out

    # handle grayscale (2D) or RGB (3D) images
    if len(input_image.shape) == 2:
        input_rows, input_cols = input_image.shape
        channels = 1
    else:
        input_rows, input_cols, channels = input_image.shape

    row_scale = input_rows / output_rows
    col_scale = input_cols / output_cols

    # create empty output
    if channels == 1:
        out = np.zeros((output_rows, output_cols))
    else:
        out = np.zeros((output_rows, output_cols, channels))

    # nearest neighbor interpolation
    for i in range(output_rows):
        for j in range(output_cols):
            orig_i = int(i * row_scale)
            orig_j = int(j * col_scale)
            if channels == 1:
                out[i, j] = input_image[orig_i, orig_j]
            else:
                out[i, j] = input_image[orig_i, orig_j]

    return out

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """
    out = 0.299 * input_image[:, :, 0] + 0.587 * input_image[:, :, 1] + 0.114 * input_image[:, :, 2]

    return out

def binary(grey_img, threshold=128):
    """Convert a greyscale image to a binary mask with threshold.

                    x_out = 0, if x_in < threshold
                    x_out = 1, if x_in >= threshold

    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        threshold (float): The threshold used for binarization, and the value range of threshold is from 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    
    return (grey_img >= threshold).astype(int)

def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    
    ### YOUR CODE HERE

    # height and width of image and kernel
    H_img, W_img = image.shape
    H_ker, W_ker = kernel.shape

    # padding size
    pad_h = H_ker // 2 
    pad_w = W_ker // 2  

    # pad image with zeroes
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # empty array for out
    out = np.zeros((H_img, W_img))

    # flip the kernel
    flipped_ker = np.flip(kernel)

    # do the convolooshun
    for i in range (H_img):
        for j in range (W_img):
            # extract the region from the padded image
            region = padded_img[i:i+H_ker, j:j+W_ker]

            # Perform element-wise multiplication and sum result
            out[i, j] = np.sum(region * flipped_ker)

    return out

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(test_img, cmap='gray')
    axes[0].set_title("original image")
    axes[1].imshow(test_output, cmap='gray')
    axes[1].set_title("convoluted image")
    plt.show()

    print("Convolved Output:\n", test_output)

def test_convRGB(image):
    """Test convolution on an RGB image."""
    img = load(image)

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])  # edge detection filter

    output = conv(img, kernel)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("original image")
    axes[1].imshow(np.clip(output, 0, 1))  # ensure values are between 0 and 1
    axes[1].set_title("filtered image")
    plt.show()


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    ### YOUR CODE HERE

    # if img is greyscale apply conv2d directly
    if len(image.shape) == 2:
        return conv2D(image, kernel)
    
    # if image is RGB apply conv2D to each channel
    channels = []
    for c in range(3):  # loop over rgb channels
        channel_out = conv2D(image[:, :, c], kernel)
        channels.append(channel_out)

    # stack processed channels back into an RGB image
    out = np.stack(channels, axis=2)

    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def gaussian_filter(image, size=5, sigma=1.0):
    """Applies Gaussian filtering to an image.
    
    input:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi).
        size: int, filter size (e.g., 3, 5, 7, must be odd).
        sigma: float, standard deviation of the Gaussian distribution.

    return:
        np.ndarray: Filtered image.
    """

    # generate gaussian kernel
    gauss_k = gauss2D(size, sigma)

    # apply convolution using existing conv() function
    filtered_img = conv(image, gauss_k)

    return filtered_img

def test_gaussian(image):
    """Test Gaussian filtering on an image."""
    img = load(image)

    # apply Gaussian filter with different settings
    blurred1 = gaussian_filter(img, size=3, sigma=0.5)
    blurred2 = gaussian_filter(img, size=5, sigma=1.0)
    blurred3 = gaussian_filter(img, size=7, sigma=2.0)

    # display results
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("original image")
    axes[1].imshow(np.clip(blurred1, 0, 1))
    axes[1].set_title("3x3, sigma=0.5")
    axes[2].imshow(np.clip(blurred2, 0, 1))
    axes[2].set_title("5x5, sigma=1.0")
    axes[3].imshow(np.clip(blurred3, 0, 1))
    axes[3].set_title("7x7, sigma=2.0")

    plt.show()

def vis_loss(image):
    img = load(image)
    blurred = gaussian_filter(img, size=5, sigma=1.0)

    diff = img - blurred

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("original image")
    axes[1].imshow(np.clip(blurred, 0, 1))
    axes[1].set_title("blurred image")
    axes[2].imshow(np.clip(diff + 0.5, 0, 1))
    axes[2].set_title("lost details")

sobel_x = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

def sobel_filter(image):
    """Applies Sobel edge detection to an image.
    
    input:
        image: numpy array of shape (Hi, Wi), must be grayscale.

    return:
        G: np.ndarray, gradient magnitude image.
        Gx: np.ndarray, gradient in X direction.
        Gy: np.ndarray, gradient in Y direction.
    """

    # check grayscale input
    if len(image.shape) == 3:
        image = greyscale(image)

    # apply Sobel filters
    Gx = conv2D(image, sobel_x)  # vertical edges
    Gy = conv2D(image, sobel_y)  # horizontal edges

    # compute gradient mag
    G = np.sqrt(Gx**2 + Gy**2)
    G = G / np.max(G)  # normalize to [0, 1] for display

    return G, Gx, Gy

def test_sobel(image):
    img = load(image)

    G, Gx, Gy = sobel_filter(img)

    # display results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("original image")
    axes[1].imshow(Gx, cmap="gray")
    axes[1].set_title("sobel X (vertical edges)")
    axes[2].imshow(Gy, cmap="gray")
    axes[2].set_title("sobel Y (horizontal edges)")
    axes[3].imshow(G, cmap="gray")
    axes[3].set_title("gradient magnitude")

    plt.show()

def downsample(image, scale=0.5):
    """Reduces an image to (scale * height, scale * width) using nearest-neighbor."""

    h, w, _ = image.shape
    new_h, new_w = int(h * scale), int(w * scale)
    return resize(image, new_h, new_w)

def blur_downsample(image, size=5, sigma=1.0, scale=0.5):
    """Applies Gaussian blur before downsampling to reduce aliasing."""
    blurred = gaussian_filter(image, size, sigma)
    return downsample(blurred, scale)

def test_downsampling(image):
    img = load(image)
    
    downsampled = downsample(img)
    blurred_downsampled = blur_downsample(img)

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("original image")
    axes[1].imshow(downsampled)
    axes[1].set_title("downsampled image")
    axes[2].imshow(blurred_downsampled)
    axes[2].set_title("blurred + downsampled image")

    plt.show()

def gaussian_pyramid(image, levels=4, size=5, sigma=1.0):
    """Generates a Gaussian pyramid with multiple levels."""
    pyramid = [image]
    for _ in range(levels):
        blurred = gaussian_filter(pyramid[-1], size, sigma)  # smooth before downsampling
        downsampled = downsample(blurred)
        pyramid.append(downsampled)
    return pyramid

def display_pyramid(image):
    img = load(image)
    pyramid = gaussian_pyramid(img)

    fig, axes = plt.subplots(1, len(pyramid), figsize=(20, 5))
    for i, test in enumerate(pyramid):
        axes[i].imshow(test)
        axes[i].set_title(f"Level {i}")

    plt.show()
    
def LoG2D(size, sigma):

    """
       Creates a Laplacian of Gaussian (LoG) Filter
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing LoG filter
    """



    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    # computer LoG kernel
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian /= 2 * np.pi * sigma**2  # normalize Gaussian

    laplacian = ((x**2 + y**2 - 2 * sigma**2) / (sigma**4)) * gaussian
    log_filter = laplacian - laplacian.mean()  # normalize to zero mean

    return log_filter

def test_log(image, size=7, sigma=1.5):
    img = load(image)

    grey = greyscale(img)
    log_kernel = LoG2D(size, sigma)
    filtered = conv2D(grey, log_kernel)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(grey, cmap='gray')
    plt.title('grayscale image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title(f'LoG filtered image\n(size={size}, sigma={sigma})')
    plt.axis('off')
    plt.show()

def test_log_variations(image):
    img = load(image)
    grey = greyscale(img)
    sizes = [5, 7, 9]
    sigmas = [1.0, 2.0, 3.0]

    fig, axes = plt.subplots(len(sizes), len(sigmas), figsize=(15, 9))

    for i, size in enumerate(sizes):
        for j, sigma in enumerate(sigmas):
            log_kernel = LoG2D(size, sigma)
            filtered = conv2D(grey, log_kernel)
            axes[i, j].imshow(filtered, cmap='gray')
            axes[i, j].set_title(f'size={size}, sigma={sigma}')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def test_log_multiscale(image):
    img = load(image)
    grey = greyscale(img)
    scales = [1.0, 0.75, 0.5, 0.25]
    sigma = 2.0
    size = 7

    fig, axes = plt.subplots(1, len(scales), figsize=(18, 5))

    for i, scale in enumerate(scales):
        scaled_img = resize(grey, int(grey.shape[0] * scale), int(grey.shape[1] * scale))
        log_kernel = LoG2D(size, sigma)
        filtered = conv2D(scaled_img, log_kernel)

        axes[i].imshow(filtered, cmap='gray')
        axes[i].set_title(f'scale {scale}')
        axes[i].axis('off')

    plt.show()


def test_log_rotation(image):

    img = load(image)
    grey = greyscale(img)
    rotated = rotate(grey, 90)
    log_kernel = LoG2D(7, 2.0)

    filtered_orig = conv2D(grey, log_kernel)
    filtered_rot = conv2D(rotated, log_kernel)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(grey, cmap='gray')
    axes[0].set_title("original Grayscale")
    axes[0].axis('off')

    axes[1].imshow(rotated, cmap='gray')
    axes[1].set_title("rotated 90°")
    axes[1].axis('off')

    axes[2].imshow(filtered_rot, cmap='gray')
    axes[2].set_title("LoG on rotated")
    axes[2].axis('off')

    plt.show()



