# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf
from tensorflow.keras.layers import Conv1D,Flatten,Dense,MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

    x_train = x_train/255.
    x_test = x_test/255.
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Conv1D(filters = 128,kernel_size=4,padding='same',input_shape = (28,28)))
    model.add(Flatten())
    model.add(Dense(2048,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    model.fit(x_train,y_train,validation_data = (x_test,y_test),epochs = 1000,batch_size=128)

    # YOUR CODE HERE
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("./tf_certificate/tf_certificate/Category2/mymodel.h5")

'''
469/469 [==============================] - 4s 9ms/step - loss: 0.0541 - accuracy: 0.9824 - val_loss: 0.6020 - val_accuracy: 0.9013
'''