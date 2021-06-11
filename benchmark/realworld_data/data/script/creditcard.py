import pandas as pd
import os
import requests

def preprocess_creditcard():
    def get_data():
        link="https://www.openml.org/data/get_csv/1673544/phpKo8OWT"
        r = requests.get(link)
        with open('./raw_data/openml_creditcard.csv', 'wb') as f:
                f.write(r.content)

    get_data()
    df = pd.read_csv("./raw_data/openml_creditcard.csv")
    # drop nan and str columns
    df = df.dropna()
    #df = df.drop(columns=['Time'])
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    #df['Class'] = df['Class'].map({0:"nominal", 1: "anomaly"})
    #df = df.sample(frac=0.025, replace=False, random_state=1)
    df = df.sort_values(by=['Time'])
    df = df.drop(columns=['Time'])
    df['Class'] = df['Class'].str.replace(r'\'', '').astype(int)
    df.to_csv("../creditcard.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    preprocess_creditcard()
