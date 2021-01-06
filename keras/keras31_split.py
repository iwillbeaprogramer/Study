import numpy as np
a = np.array(range(1,11))
size = 5

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a,size)
print("======================")
print(dataset)