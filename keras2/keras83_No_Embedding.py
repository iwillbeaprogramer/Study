from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요.','최고에요','참 잘 만든 영화에요','추천하고 싶은 영화입니다.','한 번 더 보고 싶어요','글세요','별로에요','생각보다 지루해요','연기가 어색해요','재미없어요.','너무 재미없다.',
    '참 재밋네요','규현이가 잘생길긴 했어요.']
label = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x,padding='pre',maxlen=5)
print(pad_x)
print(pad_x.shape)
print(np.unique(pad_x))
print(len(np.unique(pad_x)))
pad_x = pad_x.reshape(13,5,1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM,Flatten,Conv1D

model = Sequential()
# model.add(Embedding(input_dim=28, output_dim = 10,input_length=5)) #output_dim 은 아무거나 input_length = max_len
model.add(LSTM(64,input_shape=(5,1)))
# model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
# model.add(Flatten())
# model.add(Dense(1))
# model.summary()
model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
model.fit(pad_x,label,epochs=30,batch_size=2)
acc = model.evaluate(pad_x,label)[1]
print("acc : ",acc)

'''
Dense Only : 0.8
LSTM Only : 0.92
'''