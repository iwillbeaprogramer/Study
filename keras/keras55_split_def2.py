import numpy as np
import pandas as pd
def split_x(data, x_row, x_col, y_row, y_col):
    a, b =[], []
    x_time, y_time = 0, 0
    for i in range(data.shape[0] - x_row + 1):
        x_time += 1
        a.append(np.array(data.iloc[i:i+x_row,:x_col]))

    for i in range(data.shape[0] - y_row + 1):
        y_time += 1
        b.append(np.array(data.iloc[i:i+y_row,-y_col:]))
    
    if x_row != y_row:
        a = np.array(a)[-min(x_time,y_time):]
        b = np.array(b)[-min(x_time,y_time):]

    return  a, b



x_row, x_col = 2, 3
y_row, y_col = 2, 6
a = np.array([[1,2,3,4,5,6,7],[2,3,4,5,6,7,80],[3,4,5,6,7,8,9],[4,5,6,7,8,9,10],[5,6,7,8,9,10,11],[6,7,8,9,10,11,12]])
(X, Y) = split_xy(a, x_row, x_col, y_row, y_col)
print(X)
print(Y)
print(X.shape, Y.shape)
print(a)
