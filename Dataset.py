import tensorflow as tf
import os 
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


is_train = True
txt_path = "./VOCdevkit/VOC2012/ImageSets/Segmentation/{}".format('train.txt' if is_train else 'val.txt')

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def pascal_palette():
    palette = {(0, 0, 0): 0,
               (128, 0, 0): 1,
               (0, 128, 0): 2,
               (128, 128, 0): 3,
               (0, 0, 128): 4,
               (128, 0, 128): 5,
               (0, 128, 128): 6,
               (128, 128, 128): 7,
               (64, 0, 0): 8,
               (192, 0, 0): 9,
               (64, 128, 0): 10,
               (192, 128, 0): 11,
               (64, 0, 128): 12,
               (192, 0, 128): 13,
               (64, 128, 128): 14,
               (192, 128, 128): 15,
               (0, 64, 0): 16,
               (128, 64, 0): 17,
               (0, 192, 0): 18,
               (128, 192, 0): 19,
               (0, 64, 128): 20}

    return palette

class Dataset():
    def __init__(self,txt_path,img_path = None,label_path = None):
        self.txt = txt_path
        self.img_path = img_path
        self.label_path = label_path
        self.image_array,self.label_array ,self.classmap_array= self.read_jpg()

    def read_jpg(self):
        image_list = []
        label_list = []
        classmap_list = []

        with open(txt_path) as file:
            for line in file:
                line = line.strip('\n')
                print(line)
                image = tf.io.read_file("./VOCdevkit/VOC2012/JPEGImages/{}.jpg".format(line))
                label = tf.io.read_file("./VOCdevkit/VOC2012/SegmentationClass/{}.png".format(line))

                image = tf.image.decode_jpeg(image)
                label = tf.image.decode_png(label)

                image = tf.image.resize(image,(128,128))/255
                label = tf.image.resize(label,(128,128))

                image = np.asarray(image)
                label = np.asarray(label)

                image_list.append(image)
                label_list.append(label)

        image_array = np.array(image_list)
        label_array = np.array(label_list)
        np.save("./VOCdevkit/VOC2012/jpg.npy",image_array)
        np.save("./VOCdevkit/VOC2012/label.npy",label_array)
        #classmap_array = self.convert_from_color_segmentation(label_array)
        classmap_array = np.load("./VOCdevkit/VOC2012/classmap.npy",allow_pickle=True)
        #classmap_array = classmap_array.astype(np.float32)
        return image_array , label_array, classmap_array

    def show_img(self):
        image = self.image_array[:5]
        label = self.label_array[:5]
        plt.figure(figsize=(8,8))
        for n, image in enumerate(image):
            plt.subplot(5,2,2*n+1)
            plt.imshow(image/255)
            plt.subplot(5,2,2*n+2)
            plt.imshow(label[n])
            #plt.imshow(label.astype())
        plt.show()

    def convert_from_color_segmentation(self,label_array):
        classmap_list = []
        for arr_3d in label_array:
            arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
            palette = pascal_palette()

            for c, i in palette.items():
                m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
                arr_2d[m] = i
            classmap_list.append(arr_2d)
        classmap_array = np.asarray(classmap_list)
        np.save("./VOCdevkit/VOC2012/classmap.npy",classmap_array)
        return classmap_array

if __name__ == "__main__":
    print(os.getcwd())
    dataset = Dataset(txt_path)   
    #dataset.show_img() 
    

