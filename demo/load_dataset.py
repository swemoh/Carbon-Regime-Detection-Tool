import xarray as xr
import pandas as pd
from glob import glob
import numpy as np

def round_nav_lat(df):
    '''
    Round up the coordinates to 2 decimal places
    '''
    df['nav_lat'] = df['nav_lat'].apply(lambda x:round(x,2))
    df['nav_lon'] = df['nav_lon'].apply(lambda x:round(x,2))
    return df

def get_year_month(df, yrmonth):
    df['time_counter'] = df['time_counter'].astype("string")
    df = df.loc[df['time_counter'].str.contains(yrmonth, case=False)]
    return df.reset_index()

def get_ocean_data(year, month):

    # Read the pickle file
    data_df = pd.read_pickle(f"data/data_df_{year}.pkl")

    # Select DICP above 1500
    data_df = data_df.loc[data_df['DICP'] >= 1500]
    # Select ALK above 1700
    data_df = data_df.loc[data_df['ALK'] >= 1700]
    # Select SST below 40
    data_df = data_df.loc[data_df['sosstsst'] <= 40]

    ## Round up the coordinates
    data_df = round_nav_lat(data_df)

    months_dict = {
            'jan':'-01-',
            'feb':'-02-',
            'mar':'-03-',
            'apr':'-04-',
            'may':'-05-',
            'jun':'-06-',
            'jul':'-07-',
            'aug':'-08-',
            'sep':'-09-',
            'oct':'-10-',
            'nov':'-11-',
            'dec':'-12-',}

    ## Get the monthly data.
    yrmonth = str(year) + months_dict[month]
    df_month = get_year_month(df = data_df , yrmonth = yrmonth)

    ## Zero mean and std. dev = 1
    df_month['sst'] = (df_month['sosstsst']-df_month['sosstsst'].mean())/df_month['sosstsst'].std()
    df_month['dic'] = (df_month['DICP']-df_month['DICP'].mean())/df_month['DICP'].std()
    df_month['alk'] = (df_month['ALK']-df_month['ALK'].mean())/df_month['ALK'].std()
    df_month['fco2'] = (df_month['fco2_pre']-df_month['fco2_pre'].mean())/df_month['fco2_pre'].std()

    ## Select columns for fitting the linear model
    model_df = pd.DataFrame()
    model_df['nav_lat'] = df_month['nav_lat']
    model_df['nav_lon'] = df_month['nav_lon']
    model_df['sst'] = df_month['sst']#.apply(lambda x:round(x,3))
    model_df['dic'] = df_month['dic']#.apply(lambda x:round(x,3))
    model_df['alk'] = df_month['alk']#.apply(lambda x:round(x,3))
    model_df['fco2'] = df_month['fco2']#.apply(lambda x:round(x,3))
    model_df['sosstsst'] = df_month['sosstsst']#.apply(lambda x:round(x,3))
    model_df['DIC'] = df_month['DICP']#.apply(lambda x:round(x,3))
    model_df['ALK'] = df_month['ALK']#.apply(lambda x:round(x,3))
    model_df['fco2_pre'] = df_month['fco2_pre']#.apply(lambda x:round(x,3))

    return model_df

