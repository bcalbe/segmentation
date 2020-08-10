import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import model as M
import Dataset as D

is_train = True
txt_path = "./VOCdevkit/VOC2012/ImageSets/Segmentation/{}".format('train.txt' if is_train else 'val.txt')


def train(model,train_data,train_label):
    def scheduler(epoch,lr):
            if epoch <10:
                return lr
            elif epoch <20:
                return lr/10
            else:
                return lr/1000 
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                #tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 1e-3,patience = 2, verbose = 1)
                ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
    #loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    history = model.fit(train_data,train_label,batch_size = 4,epochs = 20,callbacks = callback,validation_split = 0.1)

def test(model,data):
    predict = model.predict(data)
    predict = np.array(predict)
    np.save("./VOCdevkit/VOC2012/predict.npy",predict)

def vis(image):
    predicts = np.load("./VOCdevkit/VOC2012/predict.npy",allow_pickle=True)
    predicts = predicts.argmax(axis = -1)
    predicts = class2png(predicts)
    plt.figure(figsize=(8,8))
    for n,predict in enumerate(predicts):
        plt.subplot(5,2,2*n+1)
        #predict = predict.argmax(axis = -1)
        predict = predict.astype(np.uint8)
        plt.imshow(predict)
        plt.subplot(5,2,2*n+2)
        plt.imshow(image[n])
        
    plt.show()

def class2png(masks):
    img_list = []
    palette = D.pascal_palette()
    for mask in masks:
        seg_img = np.zeros((128, 128,3))
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
    vis(train_dataset.image_array[2:7])
    model = M.My_model()
    #model.build(input_shape = (None,128,128,3))
    train(model.Unet,train_dataset.image_array,train_dataset.classmap_array[:,:,:,np.newaxis])
    test(model.Unet,train_dataset.image_array[2:7])


