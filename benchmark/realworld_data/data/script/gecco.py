import pandas as pd
import os
import requests

def preprocess_gecco():
    def get_data():
        link= "https://zenodo.org/record/3884398/files/1_gecco2018_water_quality.csv?download=1"
        r = requests.get(link)
        with open('./raw_data/gecco.csv', 'wb') as f:
                f.write(r.content)
    get_data()
    df = pd.read_csv("./raw_data/gecco.csv")
    # drop nan and str columns
    df = df.dropna()
    df = df.drop(columns=['Time', df.columns[0]])
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df['EVENT'] = df['EVENT'].map({False:"0", True: "1"})
    df = df.rename(columns={"EVENT": "label"})
    #df['Class'] = df['Class'].map({0:"nominal", 1: "anomaly"})
    #df = df.sample(frac=0.025, replace=False, random_state=1)
    #df = df.sort_values(by=['Time'])
    #df = df.drop(columns=['Time'])
    df.to_csv("../water_quality.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    preprocess_gecco()
