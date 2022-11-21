import os
import pandas as pd
from sklearn.model_selection import train_test_split
dataset_path = "datasets/anomaly/raw_data/yahoo_sub_5.csv"
df = pd.read_csv(dataset_path)


train_data = df[:int(0.6*len(df))]
test_data = df[~df.index.isin(train_data.index)]

print("===========train data")
print(train_data)
print("===========test_data")
print(test_data)