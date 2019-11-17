import cv2
import numpy as np
# from PIL import Image
from matplotlib import pyplot as plt
import math


class Preprocessing:
    def __init__(self):
        pass

    # this method reads an image and preprocess it by rescaling it to 150 x 100 a applying CLAHE over the image
    def read_image(self):

        image = cv2.imread("ear2.jpg")
        # image = Image.open("ear3.png").convert('RGB')
        # open_cv_image = np.array(image)
        # Convert RGB to BGR
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # plt.imshow(image)
        # plt.show()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, dsize=(100, 150), interpolation=cv2.INTER_NEAREST)
        gray_ = gray
        mean = cv2.mean(gray)[0]
        variance = np.var(gray)
        # print(gray)
        m_t = 100
        v_t = 100
        for i in range(150):
            for j in range(100):
                beta = math.sqrt(v_t * math.pow(gray[i][j] - mean, 2) / variance)
                if gray[i][j] > mean:
                    gray_[i][j] = m_t + beta
                else:
                    gray_[i][j] = m_t - beta
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray_)
        return cl1

    # in this function we are creating sw of size 50 x 50 with step size 11 so total 50 SWs
    def sub_window_creation(self, image, kernels):
        for i in range(0, 100, 11):
            for j in range(0, 50, 11):
                gabored_image = Preprocessing.process(self, image[i:i+50, j:j+50], kernels)
                # print(gabored_image.shape)
                plt.imshow(image[i:i+50, j:j+50], cmap='gray')
                plt.show()
                plt.imshow(gabored_image, cmap='gray')
                plt.show()

    # creating gabor kernel bank
    def gabor_filter(self):
        kernels = []
        for theta in [0, 45, 90, 180]:
            for sigma in [5, 10, 15, 20]:
                kernel = np.real(cv2.getGaborKernel((12, 12), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F))
                # print(kernel.shape)
                # print(kernel)
                # kernels = np.append(kernels, kernel, axis=0)
                kernels.append(kernel)
        print(len(kernels[0][0]))
        return kernels

    def process(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
        return accum


obj = Preprocessing()
processed_image = obj.read_image()
kernel_bank = obj.gabor_filter()
obj.sub_window_creation(processed_image, kernel_bank)


