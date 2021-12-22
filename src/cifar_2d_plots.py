import cifar10
import numpy as np
from scipy.fftpack import fftn,fftshift,ifftn
import cv2
import os
from skimage import color
import matplotlib.pyplot as plt

def real_imag_split(matrix):
    real = np.empty(shape=(32,32,3))
    imag = np.empty(shape=(32,32,3))
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            for c in range(matrix.shape[2]):

                real[x,y,c] = matrix[x,y,c].real
                imag[x,y,c] = matrix[x,y,c].imag

    return real,imag

def inverse_transform(fft):
    inverse_image = np.empty(shape=(32,32,3))
    for c in range(fft.shape[2]):
        inverse_image[:,:,c] = np.abs(ifftn(fft[:,:,c]))
    return inverse_image

def freq_domain(fft):
    freq = np.empty(shape=(32,32,3))
    for c in range(fft.shape[2]):
        freq[:,:,c] = np.abs(fftshift(fft[:,:,c]))
    return freq

def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(folder)

def get_histogram_gray(real,imag,img,index,folder):

    result_real = [round(item[index],2) for item in real]
    result_imag = [round(item[index], 2) for item in imag]
    result_img = [round(item[index], 2) for item in img]

    fig, axes = plt.subplots(nrows = 3, ncols = 1)

    axes[0].set_title(f'Histogram of Real part for frequency {index}')
    axes[0].hist(result_real, bins='auto', color='blue')
    axes[0].legend(labels = ['real'],loc=1, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)


    axes[1].set_title(f'Histogram of Imaginary part for frequency {index}')
    axes[1].hist(result_imag, bins='auto', color ='orange')
    axes[1].legend(labels=['imag'], bbox_to_anchor=(1.05,1.0),borderaxespad = 0)

    axes[2].set_title(f'Histogram for series at time {index}')
    axes[2].hist(result_img, bins='auto', color = 'green')
    axes[2].legend(labels=['series'], bbox_to_anchor=(1.05,1.0),borderaxespad = 0)

    fig.tight_layout()
    plt.savefig(f'{folder}/Histogram for {index}_gray.png')

def get_histogram(real,imag,img,index,folder):

    result_real = [(round(item.reshape(32*32,3)[index, 0], 2), round(item.reshape(32*32,3)[index, 1], 2), round(item.reshape(32*32,3)[index, 2], 2))
                   for item in real]
    result_imag = [(round(item.reshape(32*32,3)[index, 0], 2), round(item.reshape(32*32,3)[index, 1], 2), round(item.reshape(32*32,3)[index, 2], 2))
                   for item in imag]
    result_img = [(round(item.reshape(32*32,3)[index, 0], 2), round(item.reshape(32*32,3)[index, 1], 2), round(item.reshape(32*32,3)[index, 2], 2))
                  for item in img]

    result_real = np.array(result_real)
    result_imag = np.array(result_imag)
    result_img = np.array(result_img)


    fig, axes = plt.subplots(3,3,figsize=(30, 8))


    axes[0,0].set_title(f'Histogram of Real part for frequency {index} (channel r)')
    axes[0,0].hist(result_real[:,0], bins='auto', color='blue')
    axes[0,0].legend(labels = ['real_r'],loc=1, bbox_to_anchor=(1.05,1.0),borderaxespad = 0)

    axes[1,0].set_title(f'Histogram of Imaginary part for frequency {index} (channel r)')
    axes[1,0].hist(result_imag[:,0], bins='auto', color ='orange')
    axes[1,0].legend(labels=['imag_r'], bbox_to_anchor=(1.05,1.0),borderaxespad = 0)

    axes[2,0].set_title(f'Histogram for series at time {index} (channel r)')
    axes[2,0].hist(result_img[:,0], bins='auto', color = 'green')
    axes[2,0].legend(labels=['series_r'], bbox_to_anchor=(1.05,1.0),borderaxespad = 0)

    axes[0, 1].set_title(f'Histogram of Real part for frequency {index} (channel g)')
    axes[0, 1].hist(result_real[:, 1], bins='auto', color='blue')
    axes[0, 1].legend(labels=['real_g'], loc=1, bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

    axes[1, 1].set_title(f'Histogram of Imaginary part for frequency {index} (channel g)')
    axes[1, 1].hist(result_imag[:, 1], bins='auto', color='orange')
    axes[1, 1].legend(labels=['imag_g'], bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

    axes[2, 1].set_title(f'Histogram for series at time {index} (channel g)')
    axes[2, 1].hist(result_img[:, 1], bins='auto', color='green')
    axes[2, 1].legend(labels=['series_g'], bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

    axes[0, 2].set_title(f'Histogram of Real part for frequency {index} (channel b)')
    axes[0, 2].hist(result_real[:, 2], bins='auto', color='blue')
    axes[0, 2].legend(labels=['real_b'], loc=1, bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

    axes[1, 2].set_title(f'Histogram of Imaginary part for frequency {index} (channel b)')
    axes[1, 2].hist(result_imag[:, 2], bins='auto', color='orange')
    axes[1, 2].legend(labels=['imag_b'], bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

    axes[2, 2].set_title(f'Histogram for series at time {index} (channel b)')
    axes[2, 2].hist(result_img[:, 2], bins='auto', color='green')
    axes[2, 2].legend(labels=['series_b'], bbox_to_anchor=(1.05, 1.0), borderaxespad=0)

    fig.tight_layout()
    plt.savefig(f'{folder}/Histogram for {index}.png')

def real_imag_split_gray(matrix):
    real = [item.real for item in matrix]
    imag = [item.imag for item in matrix]
    return real,imag

folder = 'results'
check_folder(folder)

imgs = []
real_parts = []
imag_parts = []

imgs_gray = []
real_parts_gray = []
imag_parts_gray = []

count = 1
for image, _ in cifar10.data_batch_generator():
    image = image / 255
    imgs.append(image)

'''
for image, _ in cifar10.data_batch_generator():
    imgGray = color.rgb2gray(image)
    gray_fft = fftn(imgGray)
    real, imag = real_imag_split_gray(gray_fft.reshape(32 * 32))
    real_parts_gray.append(real)
    imag_parts_gray.append(imag)
    imgs_gray.append(imgGray.reshape(32*32))

    image = image/255

    fft = np.empty(shape=(32, 32, 3), dtype = complex)
    for i in range(fft.shape[2]):
        fft[:,:,i]  = fftn(image[:,:,i])


    real, imag = real_imag_split(fft)
    real_parts.append(real)
    imag_parts.append(imag)
    imgs.append(image)
    freq = freq_domain(fft)
    inversed_image = inverse_transform(fft)



   '''
    if count%2000 == 0 and count <= 6000:
        cv2.imwrite(f'{folder}/Frequency domain {count}.png',freq)
        #cv2.imwrite(f'{folder}/Frequency domain {count}_gray.png',255*np.abs(fftshift(gray_fft)))
        cv2.imwrite(f'{folder}/Inversed transformation {count}.png',inversed_image*255)
        #cv2.imwrite(f'{folder}/Inversed transformation {count}_gray.png', 255*np.abs(ifftn(gray_fft)))
        cv2.imwrite(f'{folder}/Original image {count}.png',image*255)
        #cv2.imwrite(f'{folder}/Original image {count}_gray.png', 255*imgGray)
    '''

    count += 1
    print(count)

indexs = np.arange(60,900,40)

'''
for index in indexs:
    get_histogram_gray(real_parts_gray, imag_parts_gray, imgs_gray, index, folder)
    get_histogram(real_parts, imag_parts, imgs, index, folder)
'''

'''
train_x = np.array(train_x[:1000])
test_x = np.array(test_x[:1000])
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]


train_x = train_x.reshape(train_x.shape[0],28,28)
test_x = test_x.reshape(test_x.shape[0],28,28)
train_x = fftn(train_x)
test_x = fftn(test_x)

train_x = train_x.reshape(train_x.shape[0],28*28)
test_x = test_x.reshape(test_x.shape[0],28*28)
train_x = real_imag_split(train_x)
test_x = real_imag_split(test_x)
