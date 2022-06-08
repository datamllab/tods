import pymongo
import os
import pandas as pd
myclient = pymongo.MongoClient("mongodb://localhost:27017/")


def create_connection():
    return pymongo.MongoClient("mongodb://localhost:27017/")['tods']

def read_into_mongodb(connection, path, tablename):

    exist = tablename in connection.list_collection_names()

    connection[tablename].drop()
    table = connection[tablename]
    df = pd.read_csv(path)
    table.insert_many(df.to_dict('record'))

def read_from_mongodb(connection, tablename):
    table = connection[tablename]

    df = pd.DataFrame(list(table.find()))

    df = df.drop(['_id'], axis = 1)

    return df



this_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(this_path, '../../datasets/anomaly/raw_data/yahoo_sub_5.csv')


db = create_connection()

read_into_mongodb(db, path, 'yahoo_sub_5')

df = read_from_mongodb(db, 'yahoo_sub_5')
print(df)