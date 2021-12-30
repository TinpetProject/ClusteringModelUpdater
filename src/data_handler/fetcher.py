import datetime
import pandas as pd
from params import *
from utils.process_data import new_dataset_daily, new_dataset_hourly
import pymysql
import redis
import json 

def connect_mysql():
    print('Connecting...')
    conn = pymysql.connect(host=DB_HOST, 
                            user=DB_ACC, 
                            password=DB_PASS, 
                            db=DB_NAME, 
                            charset=DB_CHARSET,
                            cursorclass=pymysql.cursors.DictCursor)
                        
    return conn

def connect_redis():
    print('Connecting redis...')
    r = redis.Redis(host=REDIS_HOST, 
                    port=REDIS_PORT,
                    password=REDIS_PASSWORD)

    return r

REDIS_CLIENT = connect_redis()

def fetch_data():
    print('Fetching data...')
    conn = connect_mysql()

    with conn.cursor() as cursor:
        sql = 'SELECT * FROM `View_PetInformation`'
        cursor.execute(sql)
        result = cursor.fetchall()

        df = pd.DataFrame(result)
        df.to_csv('./data/tmp_pets.csv', index=False)
        print(result[0])

def push_data(data: dict):
    print('Pushing data...')

    for k, v in data.items():
        json_data = json.dumps(v)
        REDIS_CLIENT.set(k, json_data)
    # r_client.set('test', 'test')
    print(REDIS_CLIENT.get('7de791e4-19fd-1f90-f5e4-cbefc3792353'))

if __name__ == '__main__':
    fetch_data()