import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import model as M
import Dataset as D

is_train = True
txt_path = "./VOCdevkit/VOC2012/ImageSets/Segmentation/{}".format('train.txt' if is_train else 'val.txt')


def train(model,train_data,train_label):
    callback = [#tf.keras.callbacks.LearningRateScheduler(scheduler),
                #tf.keras.callbacks.EarlyStopping(monitor = 'loss',min_delta = 1e-3,patience = 2, verbose = 1)
                ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
    #loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    history = model.fit(train_data,train_label,batch_size = 1,epochs = 10)

def test(model,data):
    seg_img = np.zeros((128, 128,3))
    predict = model.predict(data)
    predict = np.array(predict)
    np.save("./VOCdevkit/VOC2012/predict.npy",predict)

def vis():
    predicts = np.load("./VOCdevkit/VOC2012/predict.npy",allow_pickle=True)
    plt.figure(figsize=(8,8))
    for n,predict in enumerate(predicts):
        plt.subplot(5,1,n+1)
        predict = predict.argmax(axis = -1)*255
        predict = predict.astype(np.uint8)
        plt.imshow(predict*255,cmap = 'gray')

    # plt.figure(figsize=(8,8))
    # for n, image in enumerate(image):
    #     plt.subplot(5,2,2*n+1)
    #     plt.imshow(image/255)
    #     plt.subplot(5,2,2*n+2)
    #     plt.imshow(label[n])
        #plt.imshow(label.astype())
    plt.show()



if __name__ == "__main__":
    #vis()
    train_dataset = D.Dataset(txt_path)
    model = M.My_model()
    #model.build(input_shape = (None,128,128,3))
    train(model.Unet,train_dataset.image_array,train_dataset.classmap_array[:,:,:,np.newaxis])
    model.summary()
    test(model,train_dataset.image_array[:5])


