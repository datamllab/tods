
import pandas as pd
import json


label_file = open('combined_labels.json', 'r')
label_info = json.load(label_file)

for key in label_info.keys():
    df = pd.read_csv(key) 
    fpath, fname = key.split('/')[0], key.split('/')[1]
    label = []
    for _, row in df.iterrows():
        if row['timestamp'] in list(label_info[key]):
            label.append('1')
        else:
            label.append('0')
    df['label'] = label
    df.to_csv(fpath+"/labeled_"+fname)

