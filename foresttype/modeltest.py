from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
import tensorflow as tf


#모델 구조 지정
def get_model_map(p_input_shape):
    model = Sequential()
    model.add(Dense(units=5184, activation='relu', input_shape=p_input_shape))
    model.add(BatchNormalization())
    model.add(Reshape((9,9,64)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(units=2, activation='softmax'))
    return model

def get_model_map_fun(p_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = Dense(5184, activation='relu', name='Dense1')(input_layer)
    x2 = BatchNormalization(name='BatchNormalization1')(x1)
    x3 = Reshape((9,9,64), name = 'Reshape1')(x2)
    x4 = UpSampling2D((2,2), name = 'UpSampling2D1')(x3)
    x5 = Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu', name = 'Conv2D1')(x4)
    x6 = BatchNormalization(name = 'BatchNormalization2')(x5)
    x7 = UpSampling2D((2,2), name = 'UpSampling2D2')(x6)
    x8 = Conv2D(filters=1, kernel_size=3, padding='same', activation='relu', name = 'Conv2D2')(x7)
    x9 = Flatten(name = 'Flatten')(x8)
    x10 = Dense(units=2, activation='softmax', name = 'Dense2')(x9)
    model = tf.keras.Model(inputs=input_layer, outputs=x10, name='FunctionalModel')
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
    #model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    #model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(strides=2, pool_size=(3, 3)))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=7, activation='softmax'))
    return model

def dense_layer(p_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = Dense(units=2, activation='softmax', name = 'classifier')(input_layer)
    model = tf.keras.Model(inputs = input_layer, outputs=x1, name='classifier')
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

def encoder(p_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = Dense(5184, activation='relu', name='Dense1')(input_layer)
    x2 = BatchNormalization(name='BatchNormalization1')(x1)
    x3 = Reshape((9,9,64), name = 'Reshape1')(x2)
    x4 = UpSampling2D((2,2), name = 'UpSampling2D1')(x3)
    x5 = Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu', name = 'Conv2D1')(x4)
    x6 = BatchNormalization(name = 'BatchNormalization2')(x5)
    x7 = UpSampling2D((2,2), name = 'UpSampling2D2')(x6)
    #x8 = Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu', name = 'Conv2D2')(x7)
    #x9 = BatchNormalization(name = 'BatchNormalization3')(x8)
    #x10 = UpSampling2D((2,2), name = 'UpSampling2D3')(x9)
    x11 = Conv2D(filters=1, kernel_size=3, padding='same', activation='relu', name = 'Conv2D3')(x7)
    x12 = Flatten(name = 'Flatten')(x11)
    model = tf.keras.Model(inputs=input_layer, outputs=x12, name='encoder')
    return model

def decoder(p_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = Dense(1296, activation='relu', name='Dense1')(input_layer)
    x2 = Reshape((36,36,1), name = 'Reshape1')(x1)
    x3 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid', name='Conv2D1')(x2)
    x4 = MaxPooling2D((2,2), name = 'MaxPooling2D1')(x3)
    x5 = BatchNormalization(name='BatchNormalization1')(x4)
    x6 = Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu', name = 'Conv2D2')(x5)
    x7 = MaxPooling2D((2,2), name = 'MaxPooling2D2')(x6)
    x8 = Flatten(name = 'Flatten')(x7)
    x9 = BatchNormalization(name = 'BatchNormalization2')(x8)
    x10 = Dense(18, activation='relu', name='Dense2')(x9)
    model = tf.keras.Model(inputs=input_layer, outputs=x10, name='decoder')
    return model

def autoencoder(p_input_shape, d_input_shape, encoder, decoder):
    input_layer = tf.keras.Input(shape=p_input_shape, name='inputlayer')
    x1 = encoder(input_layer)
    x2 = decoder(x1)
    model = tf.keras.Model(inputs=input_layer, outputs=x2, name='autoencoder')
    return model

def final_model(p_input_shape, mapping_block, cnn_input_shape):
    input_layer = tf.keras.Input(shape=p_input_shape, name='InputLayer')
    x1 = mapping_block(input_layer)
    x1.trainable=False
    #x2 = x1.numpy()
    cnn = get_model_cnn(cnn_input_shape)
    x2 = cnn(x1)
    model = tf.keras.Model(inputs=input_layer, outputs=x2, name='Final_Model')
    return model

#ResNet##############################################################
def ResNet50C(p_input_shape, mapping_block, classes = 4):
    X_input = tf.keras.layers.Input(p_input_shape)
    X = X_input
    #mapping block
    X = mapping_block(X)
    
    X = tf.keras.layers.Conv2D(32, (3,3), padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = convolutional_block(X, 32, (3,3)) #use conv block (?)
    X = identity_block(X, 32, (3,3))
    X = identity_block(X, 32, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 64, (3,3)) #64->128, use conv block
    X = identity_block(X, 64, (3,3))
    X = identity_block(X, 64, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 128, (3,3)) #128->256, use conv block
    X = identity_block(X, 128, (3,3))
    X = identity_block(X, 128, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = convolutional_block(X, 256, (3,3)) #256->512, use conv block
    X = identity_block(X, 256, (3,3))
    X = identity_block(X, 256, (3,3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)
    
    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(4, activation = 'softmax')(X) # ouput layer (10 class)

    model = tf.keras.models.Model(inputs = X_input, outputs = X, name = "ResNet50C")
    
    return model

def identity_block(X, filters, kernel_size):
    X_shortcut = X
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    
    # Add
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def convolutional_block(X, filters, kernel_size):
    X_shortcut = X
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)

    X_shortcut = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)
    
    # Add
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X
#############################################################################

#모델 구조 지정
def get_model_gen(p_input_shape):
    model = Sequential()
    model.add(Dense(units=5184, activation='relu', input_shape=p_input_shape))
    model.add(BatchNormalization())
    model.add(Reshape((9,9,64)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(filters=32, kernel_size = 3, padding='same', activation='relu'))
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
    model.add(Dropout(0.5))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return model