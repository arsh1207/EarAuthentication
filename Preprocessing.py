import cv2
import numpy as np
# from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import SpectralEmbedding
import math
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS



class Preprocessing:
    def __init__(self):
        pass

    # this method reads an image and preprocess it by rescaling it to 150 x 100 a applying CLAHE over the image
    def read_image(self):
        ear_pos = ['down_ear', 'front_ear', 'left_ear', 'up_ear']
        person_num = ['000', '001', '002']
        images = []
        for i in person_num:
            for j in ear_pos:
                images.append(cv2.resize(cv2.imread("EarImages/"+i+'_'+j+".jpg", cv2.IMREAD_GRAYSCALE), dsize=(100, 150), interpolation=cv2.INTER_NEAREST))
        # print(len(image))
        # print(image[11].shape)
        # image = Image.open("ear3.png").convert('RGB')
        # open_cv_image = np.array(image)
        # Convert RGB to BGR
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # plt.imshow(image)
        # plt.show()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, dsize=(100, 150), interpolation=cv2.INTER_NEAREST)
        # plt.imshow(gray)
        # plt.show()
        processed_images = []
        for i in range(len(images)):
            gray = images[i]
            mean = cv2.mean(gray)[0]
            variance = np.var(gray)
            m_t = 100
            v_t = 100
            for i in range(150):
                for j in range(100):
                    beta = math.sqrt(v_t * math.pow(gray[i][j] - mean, 2) / variance)
                    if gray[i][j] > mean:
                        gray[i][j] = m_t + beta
                    else:
                        gray[i][j] = m_t - beta
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(gray)
            processed_images.append(cl1)
        print(processed_images[0].shape)
        return processed_images

    # in this function we are creating sw of size 50 x 50 with step size 11 so total 50 SWs
    def sub_window_creation(self, images, kernels):
        gb_all_sw = []
        label = []
        for i in range(0, 100, 11):
            for j in range(0, 50, 11):
                for k in range(len(images)):
                    image = images[k]
                    sw_image = image[i:i+50, j:j+50]
                    sw_image = cv2.resize(sw_image, dsize=(12, 12), interpolation=cv2.INTER_NEAREST)
                    # print('sw size', sw_image.shape)
                    gabored_image = Preprocessing.process(self, sw_image, kernels)
                    # print('gab size', gabored_image.shape)
                    # model = SpectralEmbedding(n_components=100, n_neighbors=10)
                    # reduced_sw = model.fit_transform(gabored_image.reshape(-1, 1))
                    # print('gab size', gabored_image.reshape(1, -1).shape)
                    # gb_all_sw.append(gabored_image)
                    gb_all_sw.append(gabored_image)
                    label.append(int(k/4))
                    # print('red size', reduced_sw.reshape(-1, 1).shape)
                    # plt.imshow(image[i:i+50, j:j+50], cmap='gray')
                    # plt.show()
                    # plt.imshow(gabored_image, cmap='gray')
                    # plt.show()
        print(len(gb_all_sw))
        print(len(gb_all_sw[0]))
        # LEM demension reduction
        model = SpectralEmbedding(n_components=100, n_neighbors=10)
        # reduced_sw = model.fit_transform(gb_all_sw)
        reduced_sw = model.fit_transform(gb_all_sw)
        
        
        knn = KNeighborsClassifier(n_neighbors=5)
        sffs = SFS(knn, 
           k_features=5, 
           forward=True, 
           floating=True, 
           scoring='accuracy',
           cv=4,
           n_jobs=-1)
        sffs = sffs.fit(reduced_sw, label)
        print('\nSequential Forward Floating Selection (k=', i, '):')
        print(sffs.k_feature_idx_)
        print('CV Score:')
        print(sffs.k_score_)
        

        
#        print('final', len(reduced_sw))
#        print('final', reduced_sw[0].shape)
#        print(label)

    # creating gabor kernel bank
    def gabor_filter(self):
        kernels = []
        Lambda = 10
        psi = 0
        gamma = 0.5
        for theta in [0, 45, 90, 180]:
            for sigma in [5, 10, 15, 20]:
                kernel = np.real(cv2.getGaborKernel((12, 12), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F))
                # print(kernel.shape)
                # print(kernel)
                # kernels = np.append(kernels, kernel, axis=0)
                kernels.append(kernel)
#                 sigma_x = sigma
#                 sigma_y = float(sigma) / gamma
#                 nstds = 3
#                 xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
#                 xmax = np.ceil(max(1, xmax))
#                 ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
#                 ymax = np.ceil(max(1, ymax))
#                 xmin = -xmax
#                 ymin = -ymax
#                 y, x = np.meshgrid(np.arrange(ymin, ymax + 1), np.arrange(xmin, xmax + 1))
#                 
#                 x_theta = x * np.cos(theta) + y * np.sin(theta)
#                 y_theta = -x * np.sin(theta) + y * np.cos(theta)
#                 
#                 gb = np.exp( -.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
#                 
#                 kernels.append(gb)
                 
        return kernels

    def process(self, img, filters):
        gabored_images = np.array([])
        # accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            gabored_images = np.append(gabored_images, fimg)
        # np.maximum(accum, fimg, accum)
        return gabored_images

    def rank_calculation(predicted,actual,rank):
      total_img = len(predicted)
      true_prediction = 0;
      for i in range(total_img):
        # row_mat = predicted[i]
        d1 = np.zeros((2,len(predicted[0])))
        for j in range(len(d1[0])):
          d1[0][j]=j  
        d1[1]= predicted[i]
        ##thinking its sorted
        d = d1[:,d1[1].argsort()]
        print(d)
        for k in range(len(d[0])-1,len(d[0]) -rank -1, -1):
          if(d[0][k] == actual[i]):
            true_prediction = true_prediction+1
      percentage = (true_prediction/total_img) *100
      return percentage

obj = Preprocessing()
processed_image = obj.read_image()
# print(len(processed_image))
# print(processed_image[0].shape)
kernel_bank = obj.gabor_filter()
obj.sub_window_creation(processed_image, kernel_bank)


