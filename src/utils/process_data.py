import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from params import *

def new_dataset_hourly(envitus, fimi):
    envitus['time'] = pd.to_datetime(envitus['time'], format='%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=25200) # 7 hours
    envitus['time'] = pd.to_datetime(envitus['time'], format='%Y-%m-%d %H:%M:%S').dt.round('60min')
    
    envitus.index = pd.DatetimeIndex(envitus['time'])
    envitus.drop(['time'], axis=1, inplace=True)
    envitus = envitus.sort_index()
    
    print(envitus.head())

    fimi['time'] = pd.to_datetime(fimi['time'], format='%Y-%m-%d %H:%M:%S').dt.round('60min')
    fimi.index = pd.DatetimeIndex(fimi['time'])
    fimi.drop(['time'], axis=1, inplace=True)
    fimi = fimi.sort_index()

    print(fimi.head())

    merged = pd.merge(fimi, envitus, how='inner', on='time')
    print(merged.head())

    clean_dat = pd.DataFrame()
    clean_dat.index = merged.index
    clean_dat['PM2_5'] = merged['PM2_5_x']
    clean_dat['PM10_0'] = merged['PM10_0_x']
    # clean_dat['temp'] = merged['temperature_x']
    # clean_dat['humidity'] = merged['humidity_x']
    clean_dat['CO'] = merged['CO_x']
    clean_dat['NO2'] = merged['NO2_x']
    clean_dat['SO2'] = merged['SO2_x']
    clean_dat['PM1_0'] = merged['PM1_0']

    clean_dat['PM2_5_cal'] = merged['PM2_5_y']
    clean_dat['PM10_0_cal'] = merged['PM10_0_y']
    clean_dat['CO_cal'] = merged['CO_y']
    clean_dat['NO2_cal'] = merged['NO2_y']
    clean_dat['SO2_cal'] = merged['SO2_y']
    clean_dat['PM1_0_cal'] = merged['PM1_0']
    # clean_dat = clean_dat.dropna(axis=0)
    print(clean_dat.head())

    clean_dat.to_csv(data_path + data_hour, index=True)

def new_dataset_daily(envitus, fimi):
    envitus['time'] = pd.to_datetime(envitus['time'], format='%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=25200) # 7 hours
    envitus['time'] = pd.to_datetime(envitus['time'], format='%Y-%m-%d %H:%M:%S').dt.round('1d')
    
    envitus.index = pd.DatetimeIndex(envitus['time'])
    envitus.drop(['time'], axis=1, inplace=True)
    envitus = envitus.sort_index()
    
    print(envitus.head())

    fimi['time'] = pd.to_datetime(fimi['time'], format='%Y-%m-%d %H:%M:%S').dt.round('1d')
    fimi.index = pd.DatetimeIndex(fimi['time'])
    fimi.drop(['time'], axis=1, inplace=True)
    fimi = fimi.sort_index()

    print(fimi.head())

    merged = pd.merge(fimi, envitus, how='inner', on='time')
    print(merged.head())

    clean_dat = pd.DataFrame()
    clean_dat.index = merged.index
    clean_dat['PM2_5'] = merged['PM2_5_x']
    clean_dat['PM10_0'] = merged['PM10_0_x']
    # clean_dat['temp'] = merged['temperature_x']
    # clean_dat['humidity'] = merged['humidity_x']
    clean_dat['CO'] = merged['CO_x']
    clean_dat['NO2'] = merged['NO2_x']
    clean_dat['SO2'] = merged['SO2_x']
    clean_dat['PM1_0'] = merged['PM1_0']

    clean_dat['PM2_5_cal'] = merged['PM2_5_y']
    clean_dat['PM10_0_cal'] = merged['PM10_0_y']
    clean_dat['CO_cal'] = merged['CO_y']
    clean_dat['NO2_cal'] = merged['NO2_y']
    clean_dat['SO2_cal'] = merged['SO2_y']
    clean_dat['PM1_0_cal'] = merged['PM1_0']
    # clean_dat = clean_dat.dropna(axis=0)
    print(clean_dat.head())

    clean_dat.to_csv(data_path + data_day, index=True)

def resample_dataset():
    fimi1 = pd.read_csv("Data/fimi/envitus_fimi14.csv", header=0)
    fimi1.index = pd.DatetimeIndex(fimi1['datetime'])

    mean_fimi1 = pd.DataFrame()
    mean_fimi1.index = fimi1.index
    mean_fimi1['PM2_5'] = fimi1['PM2_5'].resample('30Min').mean().round(2)
    mean_fimi1['PM10_0'] = fimi1['PM10_0'].resample('30Min').mean().round(2)
    mean_fimi1['temp'] = fimi1['temp'].resample('30Min').mean().round(2)
    mean_fimi1['humidity'] = fimi1['humidity'].resample('30Min').mean().round(2)
    mean_fimi1['PM2_5_cal'] = fimi1['PM2_5_cal'].resample('30Min').mean().round(2)
    mean_fimi1['PM10_0_cal'] = fimi1['PM10_0_cal'].resample('30Min').mean().round(2)
    mean_fimi1['temp_cal'] = fimi1['temp_cal'].resample('30Min').mean().round(2)
    mean_fimi1['humidity_cal'] = fimi1['humidity_cal'].resample('30Min').mean().round(2)
    clean_dat = mean_fimi1.dropna(axis=0)
    print(clean_dat.head())
    clean_dat.to_csv('Data/fimi/envitus_fimi14_mean.csv', index=True)


def plot_data_new():
    print("Plotting data...")
    merged = pd.read_csv(data_path + data_day,header=0)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, sharex=True)
    ax1.plot(merged['time'],merged['PM2_5_cal'], 'b')
    # ax1.plot(merged['time'],merged['PM2_5_cal'], 'r')
    ax1.set(xlabel='Date', ylabel='PM2.5')
    ax2.plot(merged['time'],merged['SO2_cal'], 'b')
    # ax2.plot(merged['time'],merged['SO2_cal'], 'r')
    ax2.set(xlabel='Date', ylabel='SO2')
    ax3.plot(merged['time'],merged['CO_cal'], 'b')
    # ax3.plot(merged['time'],merged['CO2_cal'], 'r')
    ax3.set(xlabel='Date', ylabel='CO')
    # ax3.plot(merged['time'],merged['CO'], 'b')
    # # ax3.plot(merged['time'],merged['CO2_cal'], 'r')
    # ax3.set(xlabel='Date', ylabel='CO')
    ax4.plot(merged['time'],merged['temp'], 'b')
    # ax3.plot(merged['time'],merged['CO2_cal'], 'r')
    ax4.set(xlabel='Date', ylabel='temp')
    ax5.plot(merged['time'],merged['humidity'], 'b')
    # ax3.plot(merged['time'],merged['CO2_cal'], 'r')
    ax5.set(xlabel='Date', ylabel='humidity')
    plt.gcf().autofmt_xdate()
    # plt.show()
    plt.savefig(log_path + 'data_daily.png')

def plot_correlation():
    merged = pd.read_csv("Data/fimi/envitus_fimi14_mean.csv", header=0)
    
    corr = merged.corr()
    ans = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)

    #save image 
    figure = ans.get_figure()    
    figure.savefig('img/correlations.png', dpi=800)

if __name__ == '__main__':
    # plot_correlation()
    plot_data_new()