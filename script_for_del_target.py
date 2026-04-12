import pandas as pd

df = pd.read_csv('data/test.csv')
df = df.drop(columns=['target'])
df.to_csv('data/test.csv', index=False)