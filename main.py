import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import model as M
import Dataset as D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = InteractiveSession(config=config)
class_weight = {0:0.5 , 1:1. , 2:1. , 3:1. , 4:1. , 5:1. , 6:1. , 7:1. , 8:1. , 9:1. , 10:1. , 
11:1. , 12:1. , 13:1. , 14:1. , 15:1. , 16:1. , 17:1. , 18:1. , 19:1. , 20:1.}

is_train = True
txt_path = "./VOCdevkit/VOC2012/ImageSets/Segmentation/{}".format('train.txt' if is_train else 'val.txt')
image_size = (128,128)


def train(model,train_data,train_label):
    epoch = 5
    def scheduler(epoch,lr):
            if epoch <3:
                return lr
            elif epoch <6:
                return lr/10
            else:
                return lr/100 
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                #tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 1e-3,patience = 2, verbose = 1)
                ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3),
    #loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    history = model.fit(train_data,train_label,batch_size = 2,epochs = epoch,callbacks = callback,validation_split = 0.1)
    model.save('model/model_{}_{}.h5'.format(image_size[0],epoch))


def test(model,data):
    predict = model.predict(data)
    predict = np.array(predict)
    np.save("./VOCdevkit/VOC2012/predict_{}.npy".format(image_size[0]),predict)

def vis(image,true_mask):
    predicts = np.load("./VOCdevkit/VOC2012/predict.npy",allow_pickle=True)
    predicts = predicts.argmax(axis = -1)
    predicts = class2png(predicts)
    plt.figure(figsize=(12,12))
    for n,predict in enumerate(predicts):
        plt.subplot(10,3,3*n+1)
        #predict = predict.argmax(axis = -1)
        predict = predict.astype(np.uint8)
        plt.imshow(predict)
        plt.subplot(10,3,3*n+2)
        plt.imshow(image[n])
        plt.subplot(10,3,3*n+3)
        plt.imshow(true_mask[n]/255)
    plt.show()

def class2png(masks):
    img_list = []
    palette = D.pascal_palette()
    for mask in masks:
        seg_img = np.zeros((image_size[0], image_size[1],3),dtype = np.uint8)
        for k,v in palette.items():
            m = np.array([mask == v])
            m = np.squeeze(m)
            seg_img[:,:,0] += ( m * k[0] ).astype('uint8')
            seg_img[:,:,1] += ( m * k[1] ).astype('uint8')
            seg_img[:,:,2] += ( m * k[2] ).astype('uint8')
        img_list.append(seg_img)
    return np.array(img_list)
    



if __name__ == "__main__":
    
    train_dataset = D.Dataset(txt_path)
    vis(train_dataset.image_array[2:10],train_dataset.label_array[2:10])
    model = M.My_model()
    #model.build(input_shape = (None,128,128,3))
    train(model.Unet,train_dataset.image_array,train_dataset.classmap_array[:,:,:,np.newaxis])
    test(model.Unet,train_dataset.image_array[2:10])


