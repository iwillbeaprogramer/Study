import pandas as pd

df = pd.DataFrame([[1,2,3,4],[4,5,6,7],[7,8,9,10]],columns=list('abcd'),index=('가','나','다'))
print(df)

df2 = df

df2['a']=100
df3 = df.copy()
df2['b']=333
print('========')
print(df)
print(df2)
print(df3)