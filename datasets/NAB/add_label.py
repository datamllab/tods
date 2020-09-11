
import pandas as pd
import json
import os
import time
import datetime


label_file = open('combined_labels.json', 'r')
label_info = json.load(label_file)

for key in label_info.keys():
    df = pd.read_csv(key) 
    fpath, fname = key.split('/')[0], key.split('/')[1]
    label = []
    unix_timestamp = []
    for _, row in df.iterrows():
        if row['timestamp'] in list(label_info[key]):
            label.append('1')
        else:
            label.append('0')
        timestamp = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S').timestamp()
        unix_timestamp.append(timestamp)
    df['label'] = label
    df['timestamp'] = unix_timestamp
    df.to_csv(fpath+"/labeled_"+fname, index=False)
    #os.remove(key)

