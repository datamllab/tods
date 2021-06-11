import pandas as pd
import os
import requests

def get_data():
    link="https://bitbucket.org/gsudmlab/mvtsdata_toolkit/downloads/petdataset_01.zip"
    r = requests.get(link)
    with open('./swan_sf.zip', 'wb') as f:
            f.write(r.content)
    os.system("unzip swan_sf.zip")

def read_labeled_data():
    dir_path = "./petdataset_01"
    files = os.listdir(dir_path)
    inlier = []
    label = {}
    for f in files:
        #if "csv" not in f and "NF" not in f:
        if "csv" not in f:
            continue
        label = f.split("lab[")[1].split("]")[0]
        #print(label)
        f_path = os.path.join(dir_path, f)
        df = pd.read_csv(f_path, header=0, sep='\t')
        df['label'] = label
        inlier.append(df)
    df = pd.concat(inlier, axis=0, ignore_index=True)
    df = df.sort_values(by=['Timestamp'])
    drop_cols = [col for col in df.columns if "label" in col or "loc" in col or "Timestamp" in col][:-1]
    df = df.drop(columns=drop_cols)
    df.reset_index(drop=True, inplace=True)
    df = df.fillna(method='ffill')
    df = df.dropna(axis="columns")
    df['label'].replace({"NF":0, "C":1, "B":1, "M":1, "X":1}, inplace=True)
    df['IS_TMFI'].replace({True:1, False:0}, inplace=True)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv("../swan_sf.csv", index=False)

if __name__ == "__main__":
    get_data()
    read_labeled_data()
    os.system("rm -rf swan_sf.zip petdataset_01")
