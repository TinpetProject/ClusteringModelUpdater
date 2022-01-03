import time
from data_handler.fetcher import fetch_data, push_data, REDIS_CLIENT
from data_handler.preprocessor import preprocess_data
from trainer.trainer import train_model
from params import *
from multiprocessing import Process

def train():
    fetch_data()
    preprocess_data()
    data = train_model()
    push_data(data)

def predict_new_pet(petID):
    print(f'Predicting new pet... {petID}')

def redis_listener():
    print('REDIS SUBCRIBER')
    subcriber = REDIS_CLIENT.pubsub()
    subcriber.subscribe(['new_pet_data'])
    
    for message in subcriber.listen():
        predict_new_pet(message)

def main():
    start_time = time.time()
    first_time = True
    days_updater = 60 * 60 # 1 hour in seconds
    # days_updater = 5
    
    print('Starting...')
    p_redis_listener = Process(target=redis_listener)
    p_redis_listener.daemon = True
    p_redis_listener.start()
    print('Started...')

    while (True):
        ellapse_time = time.time() - start_time
        if ellapse_time > days_updater or first_time:
            p = Process(target=train)
            p.daemon=True
            p.start()
            p.join()
            first_time = False
            start_time = time.time()
        
        time.sleep(3600)
        start_time = time.time()

    p_redis_listener.join()

   
def test_data():
    from utils.process_data import plot_data_new
    plot_data_new()

def test_dataset():
    from utils.dataset import PredictionDataset

    dataset = PredictionDataset(data=data_path + data_hour)
    # for i in range(len(dataset)):
    in_i, out_i = dataset[0]
    print(in_i.shape)
    print(out_i.shape)

if __name__ == '__main__':
    # test_data()
    main()
    # train()
    # preprocess_data()
    # dat = train_model()
    # push_data(dat)