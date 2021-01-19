import numpy as np
def split_x(data,x_row,x_col):
    x = []
    for i in range(len(data)-x_row+1):
        subset = data[i:(i+x_row),1:x_col]
        x.append(subset)
    return np.array(x)

a = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10]])

result = split_x(a,2,3)

print(result)

    