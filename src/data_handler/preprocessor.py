import pandas as pd
import numpy as np
import datetime

def preprocess_data():
    data = pd.read_csv('./data/tmp_pets.csv', header=0)
    processed_data = pd.DataFrame()

    processed_data['petID'] = data['petID']
    processed_data['gender'] = data['gender']
    # pet DOB
    base_date = datetime.datetime(2010,1,1)
    delta_t = pd.to_datetime(data['petDOB'], format='%Y-%m-%d') - base_date
    processed_data['petDOB'] = delta_t.apply(lambda x: x.days)
    processed_data['petDOB'] = processed_data['petDOB'] / max(processed_data['petDOB'])
    # pet type
    pets = data['petType'].unique()
    pets_idx = {pet:idx for idx, pet in enumerate(pets)}
    processed_data['petType'] = data['petType'].map(pets_idx)
    # pet color
    pets = data['petColor'].unique()
    pets_idx = {pet:idx for idx, pet in enumerate(pets)}
    processed_data['petColor'] = data['petColor'].map(pets_idx)
    # pet favorite
    unq_atts = unique_value_attributes(data, 'petFavorite')
    processed_data['petFavorite'] = data['petFavorite'].apply(extract_val, args=(unq_atts,))
    # pet property
    unq_atts = unique_value_attributes(data, 'petProperty')
    processed_data['petProperty'] = data['petProperty'].apply(extract_val, args=(unq_atts,))
    
    processed_data.to_csv('./data/processed_pets.csv', index=False)

def extract_val(value, unq_atts):
    if value:
        if type(value) == str:
            keywords = value.split(', ')
            return sum([unq_atts[k] for k in keywords]) / len(keywords)
        else:
            return 0
    else:
        return 0

def unique_value_attributes(data, attribute):
    unique_strings = data[attribute].unique()
    
    actual_unique = {}
    idx = 0
    for string in unique_strings:
        if type(string) == str:
            # print(string)
            keywords = string.split(', ')
            for k in keywords:
                if k not in actual_unique.keys():
                    actual_unique[k] = idx
                    idx += 1

    return actual_unique


if __name__ == '__main__':
    # preprocess_data()
    data = pd.read_csv('./data/tmp_pets.csv', header=0)
    print(data['petType'].unique())


