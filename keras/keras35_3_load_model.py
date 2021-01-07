import numpy as np
a = np.array(range(1,11))

from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

datasets = split_x(a,5)
x = datasets[:,:-1]
x=x.reshape(-1,4,1)
y = datasets[:,-1]

y_pred = model.predict(x)
print(y_pred)