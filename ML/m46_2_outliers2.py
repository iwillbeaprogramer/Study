# 실습
# outliers1 의 함수를 다차원 버전으로 확장

import numpy as np

aaa = np.array([[1,2,3,4,6,7,90,100,5000,10000],[10,20,3,40,50,60,70,8,90,100]]).T
# print(aaa.shape)


def outliers(data_out):
    data = data_out.transpose()
    outlier = []
    for i in range(data.shape[0]):
        quartile_1, q2, quartile_3 = np.percentile(data[i], [25, 50, 75])
        # print('1사분위 :', quartile_1)
        # print('q2 :', q2)
        # print('3사분위 :', quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print(np.where((data[i] > upper_bound) | (data[i] < lower_bound)))

        outlier.append(np.where((data[i] > upper_bound) | (data[i] < lower_bound)))

    return np.array(outlier)

result = outliers(aaa)
print(result)