
from tensorflow.keras.layers import BatchNormalization, Dense, Conv2D, \
        MaxPooling2D, GlobalMaxPooling2D, Dropout, LeakyReLU, ZeroPadding2D
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


basic = False
train = False

def create_model(batch_norm):
    """Creates the CNN model
    Args:
        batch_norm (bool): decides whether to batch normalize the conv layers
    Returns:
        model (Sequential): CNN model
    """

    model = Sequential()

    model.add(ZeroPadding2D(padding=(1, 1), input_shape= (128, 43, 1),
        data_format="channels_last"))
    model.add(Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', 
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))

    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', 
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))

    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_last"))

    model.add(Dropout(.25))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding='same',
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', 
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))

    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_last"))

    model.add(Dropout(.25))

    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(128, kernel_size=3, strides=(1, 1), padding='same',
        data_format = 'channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())


    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', 
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # 3x3 max pooling layer
    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_last"))

    # 0.25 dropout layer
    model.add(Dropout(.25))

    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', 
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', 
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    model.add(GlobalMaxPooling2D(data_format="channels_last"))

    model.add(Dense(1024))
    if batch_norm: model.add(BatchNormalization())

    model.add(Dropout(.5))

    model.add(Dense(11, activation='sigmoid'))

    model.summary()

    

    return model

def image_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(128,43,1)))
    model.add(layers.Conv2D(6, 5, activation='relu'))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(16, 5, activation='relu'))
    model.add(layers.AveragePooling2D(2))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(120, 5, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(11, activation='softmax'))
    model.summary()
    return model

def base_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(128,43,1)))
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(11, activation='softmax'))
    model.summary()
    return model


def complex_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(128,43,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(layers.GlobalMaxPooling2D())
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11))
    model.add(Activation('softmax'))
    model.summary()

    return model

val_x = np.load("val_x.npy",allow_pickle=True)

val_y = np.load("val_y.npy",allow_pickle=True)
test_x =  np.load("test_x.npy",allow_pickle=True)
test_y =  np.load("test_y.npy",allow_pickle=True)
train_x = np.load("train_x.npy",allow_pickle=True)

train_y = np.load("train_y.npy",allow_pickle=True)
print(train_x.shape)

with tf.device('/device:GPU:4'):
    model = None
    epoch = 2
    opt = None
    if basic:
        model = base_model()
        opt = "adam"
    else:
        model = complex_model()
        epoch = 8
    opt = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy,metrics=['accuracy'])
    if train:
        history = model.fit(x=train_x, y=train_y, batch_size=128, epochs=epoch, validation_data=(val_x, val_y), shuffle=True)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        if basic:
            plt.savefig('base_accuracy.png')
        else:
            plt.savefig('complex_accuracy.png')
        plt.show()
        plt.cla()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        if basic:
            plt.savefig('base_loss.png')
        else:
            plt.savefig('complex_loss.png')

        plt.show()
        if basic:
            model.save_weights('models/basic_2_model')
        else:
            model.save_weights('models/complex_2_model')
    else:
        if basic:
            model.load_weights('models/basic_2_model')
        else:
            model.load_weights('models/complex_2_model')
    model.evaluate(test_x,test_y)
    y_pred = model.predict(test_x)
    prediction_dict = {name: pred for name, pred in zip(model.output_names, y_pred)}
    print(prediction_dict)

    
    y_pred = np.argmax(y_pred, axis=1)


    cm = confusion_matrix(test_y, y_pred)

    classes=np.array(['bass', 'brass', 'flute', 'guitar', 
             'keyboard', 'mallet', 'organ', 'reed', 
             'string', 'synth_lead', 'vocal'])
    classes = classes[unique_labels(test_y, y_pred)]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title = "Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ## Display the visualization of the Confusion Matrix.
    plt.show()



