from pandas.io import pickle
from sklearn.cluster import KMeans
import pandas as pd
from joblib import dump, load

def train_model():

    print('Training model...')
    data = pd.read_csv('data/processed_pets.csv', header=0)
    X = data.to_numpy()[:, 2:]

    model = KMeans(n_clusters=2, init='k-means++').fit(X)

    dump(model, 'log/model_checkpoints/model.joblib')
    
    def make_prediction():
        result = {}

        preds = model.predict(X)
        # print(preds.shape)
        cluster_0 = data[preds == 0]
        cluster_1 = data[preds == 1]

        cluter_0_male = cluster_0[cluster_0['gender']==1]['petID'].tolist()
        cluter_0_female = cluster_0[cluster_0['gender']==0]['petID'].tolist()

        cluter_1_male = cluster_1[cluster_1['gender'] == 1]['petID'].tolist()
        cluter_1_female = cluster_1[cluster_1['gender'] == 0]['petID'].tolist()

        for _, pet in cluster_0.iterrows():
            if pet['gender'] == 0:
                result[pet['petID']] = cluter_0_male
            else:
                result[pet['petID']] = cluter_0_female

        for _, pet in cluster_1.iterrows():
            if pet['gender'] == 0:
                result[pet['petID']] = cluter_1_male
            else:
                result[pet['petID']] = cluter_1_female
        
        # print(result)
        return result
    
    return make_prediction()

