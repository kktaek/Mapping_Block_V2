from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
import tensorflow as tf


def get_model_map_fun(p_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = Dense(4096, activation='relu', name='Dense1')(input_layer)
    x2 = BatchNormalization(name='BatchNormalization1')(x1)
    x3 = Reshape((8,8,64), name = 'Reshape1')(x2)
    x4 = UpSampling2D((2,2), name = 'UpSampling2D1')(x3)
    x5 = Conv2D(filters=16, kernel_size = 3, padding='same', activation='relu', name = 'Conv2D1')(x4)
    x6 = BatchNormalization(name = 'BatchNormalization2')(x5)
    x7 = UpSampling2D((2,2), name = 'UpSampling2D2')(x6)
    x8 = Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu', name = 'Conv2D2')(x7)
    #xd = Dropout(0.5)(x8)
    x9 = BatchNormalization(name = 'BatchNormalization3')(x8)
    x10 = UpSampling2D((2,2), name = 'UpSampling2D3')(x9)
    x11 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', name = 'Conv2D4')(x10)
    x12 = Flatten(name = 'Flatten')(x11)
    x13 = Dense(units=7, activation='softmax', name = 'Dense2')(x12)
    model = tf.keras.Model(inputs=input_layer, outputs=x13, name='FunctionalModel')
    return model

def get_model_cnn(p_input_shape):
    model = Sequential()
    model.add(Dense(units=1296, activation='relu', input_shape=p_input_shape))
    model.add(BatchNormalization())
    model.add(Reshape((36,36,1)))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3,3)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))
    return model

def dense_layer(p_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = Dense(units=7, activation='softmax', name = 'classifier')(input_layer)
    model = tf.keras.Model(inputs = input_layer, outputs=x1, name='classifier')
    return model

def final_model(p_input_shape, mapping_block, cnn_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = mapping_block(input_layer)
    #x2 = x1.numpy()
    cnn = get_model_cnn(cnn_input_shape)
    x2 = cnn(x1)
    model = tf.keras.Model(inputs=input_layer, outputs=x2, name='Final_Model')
    return model

def encoder(p_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = Dense(2304, activation='relu', name='Dense1')(input_layer)
    x2 = BatchNormalization(name='BatchNormalization1')(x1)
    x3 = Reshape((6,6,64), name = 'Reshape1')(x2)
    x4 = UpSampling2D((2,2), name = 'UpSampling2D1')(x3)
    x5 = Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu', name = 'Conv2D1')(x4)
    x6 = BatchNormalization(name = 'BatchNormalization2')(x5)
    x7 = UpSampling2D((2,2), name = 'UpSampling2D2')(x6)
    x8 = Conv2D(filters=16, kernel_size = 3, padding='same', activation='relu', name = 'Conv2D2')(x7)
    x9 = BatchNormalization(name = 'BatchNormalization3')(x8)
    x10 = UpSampling2D((2,2), name = 'UpSampling2D3')(x9)
    x11 = Conv2D(filters=1, kernel_size=3, padding='same', activation='relu', name = 'Conv2D3')(x10)
    x12 = Flatten(name = 'Flatten')(x11)
    model = tf.keras.Model(inputs=input_layer, outputs=x12, name='encoder')
    return model

def autoencoder_test(p_input_shape, autoencoder, cnn_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = autoencoder(input_layer)
    x1.trainable=False
    #x2 = x1.numpy()
    cnn = dense_layer(cnn_input_shape)
    x2 = cnn(x1)
    model = tf.keras.Model(inputs=input_layer, outputs=x2, name='Final_Model')
    return model

#모델 구조 지정
def get_model(p_input_shape):
    model = Sequential()
    model.add(Dense(units=400, activation='tanh', input_shape=p_input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(units=2, activation='softmax'))
    return model

#모델 구조 지정
def get_model_gen(p_input_shape):
    model = Sequential()
    model.add(Dense(units=4096, activation='relu', input_shape=p_input_shape))
    model.add(BatchNormalization())
    model.add(Reshape((8,8,64)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=16, kernel_size = 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))
    #CNN
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3,3)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    return model



#모델 구조 지정
def get_model_gen_v2(p_input_shape):
    model = Sequential()
    model.add(Dense(units=1568, activation='relu', input_shape=p_input_shape))
    model.add(BatchNormalization())
    model.add(Reshape((7,7,32)))

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))

    #CNN
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3,3)))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(units=300, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return model