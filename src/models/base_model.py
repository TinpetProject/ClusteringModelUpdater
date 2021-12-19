import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from utils.dataset import CalibDataset, PredictionDataset
from params import *

class BaseModel:
    def __init__(self, input_size=8, output_size=6, input_timestep=30, output_timestep=7, hidden_size=128, batch_size=128, num_layers=1, dropout=0, calib=False, hourly=True, **kwargs):
        """
        Model wrapper for all models
        """
        self.calib = calib
        self.hourly = hourly
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if calib:
            output_timestep = input_timestep
        self.train_loader, self.test_loader = self.preprocess_data(input_timestep, output_timestep)

        if calib:
            if hourly:
                from models.EncoderDecoder.hourly_calib import EncoderDecoder
                self.model = EncoderDecoder().to(self.device)
            else:
                from models.EncoderDecoder.daily_calib import EncoderDecoder
                self.model = EncoderDecoder().to(self.device)
        else:
            if hourly:
                from models.EncoderDecoder.hourly_predict import EncoderDecoder
                self.model = EncoderDecoder().to(self.device)
            else:
                from models.EncoderDecoder.daily_predict import EncoderDecoder
                self.model = EncoderDecoder().to(self.device)
        print(self.model)

    def preprocess_data(self, input_timestep, output_timestep):
        """
        Preprocess the data
        """
        if self.hourly:
            path = data_path + data_hour
        else:
            path = data_path + data_day
        
        if self.calib:
            dataset = CalibDataset(path, frame_length=input_timestep)
        else:
            dataset = PredictionDataset(path, in_length=input_timestep, out_length=output_timestep)
        
        train, test = dataset.get_splits()
        print(f'batch size: {self.batch_size}')
        train_dl = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test_dl = DataLoader(test, batch_size=self.batch_size, shuffle=False)

        self.mean, self.std = self.get_mean_std(train_dl)

        return train_dl, test_dl

    def get_mean_std(self,loader):
        mean = 0.0
        meansq = 0.0
        count = 0

        for data, _ in tqdm(loader):
            mean = mean + data.sum()
            meansq = meansq + (data**2).sum()
            count += np.prod(data.shape)

        total_mean = mean/count
        total_var = (meansq/count) - (total_mean**2)
        total_std = torch.sqrt(total_var)
        print("mean: " + str(total_mean))
        print("std: " + str(total_std))

        return total_mean, total_std

    def train(self):
        """
        Train the model
        """
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(500):
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                x_mean = x.mean(dim=1, keepdim=True)
                x_std = x.std(dim=1, keepdim=True)
                optimizer.zero_grad()
                y_pred = self.model(x, x_mean, x_std)
                
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()

    def evaluate(self):
        """
        Evaluate the model
        """
        
        preds = []
        actuals = []

        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            x_mean = x.mean(dim=1, keepdim=True)
            x_std = x.std(dim=1, keepdim=True)
            y_pred = self.model(x, x_mean, x_std)
            preds.append(y_pred.cpu().detach().numpy())
            actuals.append(y.cpu().detach().numpy())
        
        preds = np.vstack(preds)
        print(preds.shape)
        actuals = np.vstack(actuals)
        
        mse = np.mean(np.square(preds - actuals))
        mae = np.mean(np.abs(preds - actuals))
        mape = np.mean(np.abs((actuals - preds) / preds)) * 100

        print(f'MSE: {mse} MAE: {mae} MAPE: {mape}')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(preds[0,:,0], label='Predictions')
        ax.plot(actuals[0,:,0], label='Actuals')
        ax.legend()

        if self.calib:
            if self.hourly:
                file_name = 'model_hourly_calib.png'
            else:
                file_name = 'model_daily_calib.png'
        else:
            if self.hourly:
                file_name = 'model_hourly_predict.png'
            else:
                file_name = 'model_daily_predict.png'

        plt.savefig('log/' + file_name)

    def save(self):
        """
        Save the model to a file
        """
        if self.calib:
            if self.hourly:
                file_name = 'hourly_calib.pt'
            else:
                file_name = 'daily_calib.pt'
        else:
            if self.hourly:
                file_name = 'hourly_predict.pt'
            else:
                file_name = 'daily_predict.pt'
        
        torch.save(self.model.state_dict(), '../models/pt_files/' + file_name)
    
    def load(self):
        """
        Load the model
        """
        if self.calib:
            if self.hourly:
                file_name = 'model_hourly_calib.pt'
            else:
                file_name = 'model_daily_calib.pt'
        else:
            if self.hourly:
                file_name = 'model_hourly_predict.pt'
            else:
                file_name = 'model_daily_predict.pt'

        self.model.load_state_dict('../models/pt_files/' + file_name)

    def __str__(self):
        """
        Return a string representation of the model
        """
        return "[{}] ({})".format(self.__class__.__name__, self.model)