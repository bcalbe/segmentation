import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.initializers import Constant
#from tensorflow.nn import conv2d_transpose

image_shape = (128, 128)
def bilinear_upsample_weights(factor, number_of_classes):
	#"""初始化权重参数"""
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights


class MyModel(tf.keras.Model):
    def __init__(self, n_class):
        super().__init__()
        self.vgg16_model = self.load_vgg()
        
        self.conv_test = Conv2D(filters=n_class, kernel_size=(1, 1))  # 分类层
        self.deconv_test = Conv2DTranspose(filters=n_class, 
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, n_class)))  # 上采样层

    def call(self, input):
      x = self.vgg16_model(input)
      x = self.conv_test(x)
      x = self.deconv_test(x)
      return x

    def load_vgg(self):
        # 加载vgg16模型，其中注意input_tensor，include_top
        vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(image_shape[0], image_shape[1], 3)))
        # for layer in vgg16_model.layers[:15]:
        #   layer.trainable = False  # 不训练前15层模型
        vgg16_model.summary()
        return vgg16_model


class My_model():
    def __init__(self):
        self.model_name = "U-net"
        self.encoder = self.get_vgg()
        #self.decoder = self.get_decoder()
        self.Unet = self.get_unet()
      
    def get_vgg(self):
        # 加载vgg16模型，其中注意input_tensor，include_top
        vgg16_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(image_shape[0], image_shape[1], 3)))
        # for layer in vgg16_model.layers[:15]:
        #   layer.trainable = False  # 不训练前15层模型
        vgg16_model.summary()
        outputs_names = ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']
        layers = [vgg16_model.get_layer(name).output for name in outputs_names]
        model = tf.keras.Model(inputs = vgg16_model.input,outputs = layers)

        return model
    
    def upsample(self,filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def get_unet(self,OUTPUT_CHANNELS = 21):

        up_stack = [
                    self.upsample(512, apply_dropout=False), # (bs, 8, 8, 1024) for input_shape(128,128)
                    self.upsample(256, 3), # (bs, 16, 16, 1024)
                    self.upsample(128, 3), # (bs, 32, 32, 512)
                    self.upsample(64, 3), # (bs, 64, 64, 256)
                    ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='softmax') # (bs, 128, 128, 3)

        inputs = tf.keras.layers.Input(shape=[image_shape[0], image_shape[1], 3])
        x = inputs
        
        #downsample
        features = self.encoder(x)
        x = features[-1]
        features = reversed(features[:-1])

        for up,feature in zip(up_stack,features):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x,feature])

        x = last(x)
        return tf.keras.Model(inputs = inputs, outputs = x)
    

if __name__ == "__main__":
    # model = MyModel(21)    
    # input_shape = tf.keras.Input(shape = (128,128,3))
    # model.build()
    # model.summary()
    model = My_model()
    model.Unet.summary()
    tf.keras.utils.plot_model(model.Unet, "Unet.png", show_shapes=True)
    