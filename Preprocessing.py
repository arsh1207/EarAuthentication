import cv2
import numpy as np
# from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import SpectralEmbedding
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
        # plt.imshow(gray)
        # plt.show()
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
        gb_all_sw = []
        for i in range(0, 100, 11):
            for j in range(0, 50, 11):
                sw_image = image[i:i+50, j:j+50]
                sw_image = cv2.resize(sw_image, dsize=(12, 12), interpolation=cv2.INTER_NEAREST)
                print('sw size', sw_image.shape)
                gabored_image = Preprocessing.process(self, sw_image, kernels)
                print('gab size', gabored_image.shape)
                # print('gab size', gabored_image.reshape(1, -1).shape)
                gb_all_sw.append(gabored_image)
                # gb_all_sw.append(reduced_sw)
                # print('gab size', reduced_sw.shape)
                # plt.imshow(image[i:i+50, j:j+50], cmap='gray')
                # plt.show()
                # plt.imshow(gabored_image, cmap='gray')
                # plt.show()
        print(len(gb_all_sw[0]))
        model = SpectralEmbedding(n_components=100, n_neighbors=10)
        # reduced_sw = model.fit_transform(gabored_image.reshape(-1, 1))
        reduced_sw = model.fit_transform(gb_all_sw)
        print('final', len(reduced_sw))
        print('final', len(reduced_sw[0]))

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
        gabored_images = np.array([])
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            gabored_images = np.append(gabored_images, fimg)
        # np.maximum(accum, fimg, accum)
        return gabored_images


obj = Preprocessing()
processed_image = obj.read_image()
kernel_bank = obj.gabor_filter()
obj.sub_window_creation(processed_image, kernel_bank)


