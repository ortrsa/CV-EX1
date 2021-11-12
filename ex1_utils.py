"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206066326


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    # Checking that representation is 1 or 2
    if representation != 1 and representation != 2:
        print("pleas choose 1 for GRAY_SCALE or 2 for RGB")
        pass
    # Read the image and convert to the relevant representation
    if representation == 1:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # normal image
    norm_image = image / 255
    return norm_image


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if filename is None:
        print('filename is None')
        pass

    # Read image and convert to the relevant representation
    normal = imReadAndConvert(filename, representation)
    # if the image is gray we will set colormap to gray
    if representation == LOAD_GRAY_SCALE:
        plt.gray()
    plt.imshow(normal)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    if imgRGB.ndim != 3:
        print('Not an RGB image')
        pass
    if imgRGB is None:
        print('No image found')
        pass
    # make YIO matrix that will contain YIO values.
    YIQ = np.zeros(shape=(imgRGB.shape))
    # Separation into three matrix (R,G,B)
    red = imgRGB[:, :, 0]
    green = imgRGB[:, :, 1]
    blue = imgRGB[:, :, 2]
    # This matrix represents a linear relationship between RGB to YIQ
    YIQ[:, :, 0] = 0.299 * red + 0.587 * green + 0.114 * blue
    YIQ[:, :, 1] = 0.596 * red + (-0.275) * green + (-0.321) * blue
    YIQ[:, :, 2] = 0.212 * red + (-0.523) * green + 0.311 * blue

    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    if imgYIQ is None:
        print('No image found')
        pass
    if imgYIQ.ndim != 3:
        print('Not an YIQ image')
        pass
    # Separation into three matrix (y,i,q)
    y = imgYIQ[:, :, 0]
    i = imgYIQ[:, :, 1]
    q = imgYIQ[:, :, 2]
    # make RGB matrix that will contain RGB values.
    RGB = np.zeros(shape=(imgYIQ.shape))
    # This matrix represents a linear relationship between YIQ to RGB
    RGB[:, :, 0] = (y * 1 + (0.956 * i) + (0.620 * q))
    RGB[:, :, 1] = (y * 1 + (-0.272 * i) + (-0.647 * q))
    RGB[:, :, 2] = (y * 1 + (-1.108 * i) + (1.705 * q))
    return RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if imgOrig is None:
        print('No image found')
        pass
    # if the image is RGB we will convert to YIQ and use only y.
    flag = 0
    if imgOrig.ndim == 3:
        flag = 1
        YIQ = transformRGB2YIQ(imgOrig)
        y = YIQ[:, :, 0]
        imgOrig = y
    # denormalize imgOrig
    imgOrig = (imgOrig * 255).astype(int)
    new_im = imgOrig.copy()  # Make copy that we can work on.
    im_flat = imgOrig.flatten()  # Make the matrix to array.
    hist = np.zeros(256)
    new_hist = np.zeros(256)
    # make histogram
    for pix in im_flat:
        hist[pix] += 1
    # make cumsum
    cumsum = np.cumsum(hist)
    pixel = max(cumsum)  # all the pixel in this image

    # make LUT with new values
    lut = np.zeros(im_flat.size)
    lut = np.ceil(cumsum / pixel * 255)

    # we will mark all of the pixel in each intensity and change them to new intensity from "LUT"
    for intens in range(255):
        new_im[imgOrig == intens] = int(lut[intens])
    new_im_flat = new_im.flatten().astype(int)

    # make new hist after the change
    for n in new_im_flat:
        new_hist[n] += 1

    # if the image was RGB we will need to transform it back to RBG with the new "y".
    if flag == 1:
        YIQ[:, :, 0] = new_im / 255  # normalize new_im
        new_im = transformYIQ2RGB(YIQ)
    else:
        new_im = new_im / 25  # normalize new_im
    return new_im, hist, new_hist


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # if the image is RGB we will convert to YIQ and use only y.
    flag = 0
    if imOrig.ndim == 3:
        flag = 1
        YIQ = transformRGB2YIQ(imOrig)
        y = YIQ[:, :, 0]
        imOrig = y
    # make some empty lists for borders(z), images and errors
    qImage = []
    error = []
    z = []

    # flatten image
    imflat = (imOrig.flatten() * 255).astype(int)

    # make histogram
    hist = np.zeros(256)
    for n in imflat:
        hist[n] += 1

    # initial z to nQuant + 1 division
    for i in range(nQuant + 1):
        z.append(int((255 / nQuant) * i))

    # nIter is the number of optimization loops
    for k in range(nIter):
        newim = np.zeros(shape=imOrig.shape)  # newim will is the new image
        q = []  # List of Weighted Means
        # calculate Weighted Means between each 2 z divisions and add to q
        for i in range(nQuant):
            avg = int(np.average(range(z[i], z[i + 1]), weights=hist[z[i]: z[i + 1]]))
            q.append(avg)
        # add the new intensity to the new image by mapping from each division to the relevant weighted Mean
        for j in range(len(q)):
            newim[imOrig > z[j] / 255] = q[j]
        # improve borders by move each boundary to be in the middle of two means
        for i in range(1, len(z) - 1):
            z[i] = int((q[i - 1] + q[i]) / 2)
        # add MSE to error list
        error.append(np.sqrt((imOrig * 255 - newim) ** 2).mean())
        # add images to qImage list
        qImage.append(newim)

    # if the image was RGB we will need to transform it back to RBG with the new "y".
    if flag == 1:
        for i in range(len(qImage)):
            YIQ[:, :, 0] = qImage[i] / 255
            qImage[i] = transformYIQ2RGB(YIQ)

    return qImage, error
