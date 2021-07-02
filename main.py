from keras import Input, Model
from keras.layers import Conv2D, LeakyReLU
from keras.optimizer_v1 import Adam
from keras_contrib.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam

df = 64
img_rows = 128
img_cols = 128
channels = 3
img_shape = (img_rows, img_cols, channels)


def conv2d(layer_input, filters, f_size=4, normalization=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size,
               strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if normalization:
        d = InstanceNormalization()(d)
    return d


def build_discriminator():
    img = Input(shape=img_shape)

    d1 = conv2d(img, df, normalization=False)
    d2 = conv2d(d1, df * 2)
    d3 = conv2d(d2, df * 4)
    d4 = conv2d(d3, df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity)


optimizer = Adam(0.0002, 0.5)
ds = build_discriminator()
ds.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

ds()
