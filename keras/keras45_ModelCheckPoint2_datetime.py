import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.distribute import distributed_file_utils

import tensorflow as tf
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
        # `filepath` may contain placeholders such as `{epoch:02d}` and
        # `{mape:.2f}`. A mismatch between logged metrics and the path's
        # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, timer=datetime.datetime.now().strftime('%m%d_%H%M'), **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                        'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath






(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(-1,28,28,1)/255.


import datetime
filepath='../data/modelcheckpoint/'
filename='_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k45_", '{timer}', filename])
cp = MyModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

es = EarlyStopping(monitor='val_loss',patience=10)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(filters = 512,kernel_size = (2,2),padding='same',strides = 1,input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,validation_split = 0.2,epochs=30,verbose=1,batch_size=16,callbacks=[es,cp])

loss = model.evaluate(x_test,y_test,batch_size=16)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=-1)

for i in range(10):
    print('실제 : ',np.argmax(y_test[i]),'예상 : ',y_predict[i])
print('accuracy : ',loss[1])

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['accuracy'],marker='.',c='red')
plt.plot(hist.history['val_accuracy'],marker='.',c='blue')
plt.grid()
plt.title('Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy'],loc='upper right')
plt.show()






'''
CNN
accuracy :  0.9873999953269958
'''

