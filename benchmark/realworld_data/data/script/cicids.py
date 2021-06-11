import pandas as pd
import numpy as np
import os
import requests


def preprocess_web_attack():
    def get_data():
        link="http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"
        r = requests.get(link)
        with open('./raw_data/cicids.zip', 'wb') as f:
                f.write(r.content)
        os.system("unzip ./raw_data/cicids.zip -d ./raw_data")
        os.system("mv ./raw_data/MachineLearningCSV/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv ./raw_data/cicids.csv")
        os.system("rm -r MachineLearningCSV")
        os.system("rm cicids.zip")

    if get_data == True:
        get_data()
    #df = pd.read_csv("./raw_data/MachineLearningCSV/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    df = pd.read_csv("./raw_data/cicids.csv")

    df.replace([float('inf'), 'Infinity',''], np.nan, inplace=True)
    df = df.dropna()
    #df = df.sample(frac=0.05, replace=False, random_state=1)
    df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], infer_datetime_format=True)
    df = df.sort_values(by=[' Timestamp'])

    # drop nan and str columns
    drop_cols = list(df.columns)[0:5]
    drop_cols = list(df.columns)[0:5]
    drop_cols.append(list(df.columns)[6])
    df = df.drop(columns=drop_cols)

    # relabeing and put label in the first column
    df[' Label'] = df[' Label'].map({'BENIGN':"0", "Web Attack Brute Force": "1","Web Attack Sql Injection": "1", "Web Attack XSS": "1"})
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv("../web_attack.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    preprocess_web_attack()
