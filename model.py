
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
tf.enable_eager_execution()

def make_model(classes=11):
	model = models.Sequential()
	model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=4, strides=(2,2), padding='same', input_shape=(128,126,1), name="conv_1"))
	model.add(tf.keras.layers.BatchNormalization(name="bn_1"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="mpool_1"))
	model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), padding='valid', name="conv_2"))
	model.add(tf.keras.layers.BatchNormalization(name="bn_2"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="mpool_2"))
	model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1,1), padding='valid', name="conv_3"))
	model.add(tf.keras.layers.BatchNormalization(name="bn_3"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="mpool_3"))
	model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=(1,1), padding='valid', name="conv_4"))
	model.add(tf.keras.layers.BatchNormalization(name="bn_4"))
	model.add(tf.keras.layers.ReLU())
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="mpool_4"))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dropout(rate = 0.25))
	model.add(tf.keras.layers.Dense(500, activation='relu', kernel_regularizer='l2', name="fc_5"))
	model.add(tf.keras.layers.Dropout(rate = 0.5))
	model.add(tf.keras.layers.Dense(classes, activation=None, kernel_regularizer='l2', name="fc_6"))
	model.add(tf.keras.layers.Softmax())
	return model



def best_model():
    use_bias=True
    model = Sequential()
    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(128, 43, 1),
        data_format="channels_last"))
    model.add(Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', use_bias=use_bias,
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', use_bias=use_bias,
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_last"))
    model.add(Dropout(.25))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', use_bias=use_bias,
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', use_bias=use_bias,
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_last"))
    model.add(Dropout(.25))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', use_bias=use_bias,
        data_format = 'channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', use_bias=use_bias,
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_last"))
    model.add(Dropout(.25))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', use_bias=use_bias,
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_last"))
    model.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', use_bias=use_bias,
        data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(GlobalMaxPooling2D(data_format="channels_last"))
    model.add(Dense(1024, use_bias=use_bias))
    model.add(Dropout(.5))
    model.add(Dense(11, activation='sigmoid'))    
    return model

if __name__ == "__main__":
    train = False
    second_model = False
    bonus = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "second_model":
            second_model = True
        elif sys.argv[1] == "bonus":
            bonus = True
        elif sys.argv[1]=="train":
            train = True
        if len(sys.argv) > 2:
            if sys.argv[2] == "train":
                train = True

    save_dir = None
    close = False
    save_prev = ""
    if second_model:
        save_prev = "second_"
    elif bonus:
        save_prev = "bonus_"
    val_x = np.load(save_prev+"val_x.npy",allow_pickle=True)
    val_y = np.load(save_prev+"val_y.npy",allow_pickle=True)
    test_x =  np.load(save_prev+"test_x.npy",allow_pickle=True)
    test_y =  np.load(save_prev+"test_y.npy",allow_pickle=True)
    train_x = np.load(save_prev+"train_x.npy",allow_pickle=True)
    train_y = np.load(save_prev+"train_y.npy",allow_pickle=True)
    audio = np.load("audio.npy",allow_pickle=True)
    with tf.device('/device:GPU:4'):
        model = None
        epoch = 40
        opt = None
        if second_model:
            model = best_model()
            save_dir = 'best_model'
        elif bonus:
            model = make_model(classes=3)
            save_dir = 'bonus'
        else:
            model = make_model()
            save_dir = 'complex_model'
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        keras_callbacks   = [
        EarlyStopping(monitor='val_loss', patience=2, mode='min', min_delta=0.0001),
        ModelCheckpoint(save_dir, monitor='val_accuracy', save_best_only=True, mode='max')
        ]
        if second_model:
            model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        else:
            model.compile(optimizer=opt, loss='categorical_crossentropy',
                metrics=['accuracy'])
        if train:
            history = model.fit(x=train_x, y=train_y, batch_size=64, epochs=epoch, validation_data=(val_x, val_y), callbacks=keras_callbacks)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.savefig(save_dir+"_accuracy.png")
            plt.cla()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.savefig(save_dir+"_loss.png")
            plt.cla()

            model.save_weights("model_new/"+save_dir)
        
        else:
            model.load_weights("model_new/"+save_dir)
        model.evaluate(test_x,test_y)
        
        for _ in range(2):
            close = not close
            y_pred = model.predict(test_x)
            class_size = 10
            if bonus:
                class_size = 3
            count = 0

            arr = y_pred.tolist()
            result = []
            result_prob = []
            vals = [0] * class_size
            done = []
            count = 0
            for row in arr:
                if not bonus:
                    del row[9]    
                result.append(row.index(max(row)))
                result_prob.append(max(row))
                if close:
                    row.sort()
                    if row[-1] - row[-2] <= 10:
                        if result[-1] not in done:
                            vals[result[-1]] = count 
                    done.append(result[-1])
                    count += 1
            if not close:
                vals = [0] * class_size 
                done = []
                for n in range(len(result)):
                    if result_prob[n] > 0.9 or result_prob[n] < 0.1:
                        if result[n] not in done:
                            vals[result[n]] = n
                            done.append(result[n])
        
            content = []
            target_classes = []
            if bonus:
                target_classes = [" acoustic","electronic","synthesized" ]
            else:
                target_classes = [ "bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string","vocal" ]
            for n in range(len(vals)):
                content.append(audio[vals[n]])
            for num in range(len(content)):  
                sh = np.array(content[num], dtype=float).shape[0]
                length = sh / 2000
                timen = np.linspace(0., length, sh)
                plt.plot(timen, content[num])
                plt.title("" + target_classes[num])
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                if close:
                    plt.savefig(save_dir+" close " + target_classes[num])
                else:
                    plt.savefig(save_dir+" not close " + target_classes[num])
                plt.cla()
        y_pred = np.argmax(y_pred, axis=1)
        
        if not second_model:
            test_y = np.argmax(test_y,axis=1)
        cm = confusion_matrix(test_y, y_pred)
        if not bonus:
            target_classes = [ "bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string","ignore","vocal" ]
        classes=np.array(target_classes)
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
        plt.savefig(save_dir+'_base_confusion_matrix.png')




