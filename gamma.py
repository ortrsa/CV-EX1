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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep == 1:
        im = cv2.imread(img_path, 2)
    else:
        im = cv2.imread(img_path, 1)
    # create new window

    im = im / 255
    imcopy = im.copy()
    cv2.namedWindow('gamma correction')

    # create trackbar
    cv2.createTrackbar('gamma', 'gamma correction', 100, 200, track)

    # infinite loop press 'q' to exit
    while True:
        cv2.imshow('gamma correction', im)
        t = cv2.getTrackbarPos('gamma', 'gamma correction')
        gamma = t / 100
        im = np.power(imcopy, gamma)

        # press 'q' to exit
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break


# print current gamma
def track(x):
    print(x / 100)


def main():
    gammaDisplay('beach.jpg', 1)


if __name__ == '__main__':
    main()
