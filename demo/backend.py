import xarray as xr
import pandas as pd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import axes

import io
import base64

import numpy as np
import numpy.matlib

# For plotting maps
import os
os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
from mpl_toolkits.basemap import Basemap

import plotly.express as px

import seaborn as sns
# sns.set_theme(style="whitegrid")
# sns.set(rc={"figure.dpi":200,})
# sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
# sns.set_context('notebook')

import warnings
warnings.simplefilter('ignore')

from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from scipy import spatial

import random

# for regression P values
import statsmodels.api as sm
from sklearn_extra.cluster import KMedoids
from sklearn.feature_selection import chi2
from sklearn.metrics.pairwise import cosine_similarity

from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial import distance

from scipy import stats
from sklearn.feature_selection import chi2



#---------------------------- 1. Data Loading. ----------------------------

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

def get_ocean_data(year, month, is_natural, drivers_list, target, ice_slider):

    # Read the pickle file
    data_df = pd.read_pickle(f"data/orca025_{year}_df.pkl")

    # Select DICP above 1500
    if is_natural:
        data_df = data_df.loc[data_df['DICP'] >= 1500]
        data_df = data_df.loc[data_df[target] <= 500]
    else:
        data_df = data_df.loc[data_df['DIC'] >= 1500]
        data_df = data_df.loc[data_df[target] <= 500]
    # Select ALK above 1700
    data_df = data_df.loc[data_df['ALK'] >= 1700]
    # Select SST below 40
    data_df = data_df.loc[data_df['SST'] <= 40]

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
    # print(df_month)
    # ## Zero mean and std. dev = 1
    # df_month['sst'] = (df_month['sosstsst']-df_month['sosstsst'].mean())/df_month['sosstsst'].std()
    # df_month['dic'] = (df_month['DICP']-df_month['DICP'].mean())/df_month['DICP'].std()
    # df_month['alk'] = (df_month['ALK']-df_month['ALK'].mean())/df_month['ALK'].std()
    # df_month['fco2'] = (df_month['fco2_pre']-df_month['fco2_pre'].mean())/df_month['fco2_pre'].std()

    ## Select columns for fitting the linear model
    model_df = pd.DataFrame()
    model_df['nav_lat'] = df_month['nav_lat']
    model_df['nav_lon'] = df_month['nav_lon']
    for driver in drivers_list:
        if is_natural:
            model_df[driver] = df_month[driver]#.apply(lambda x:round(x,3))
        else:
            if driver == 'DICP':
                model_df['DIC'] = df_month['DIC']
            else:
                model_df[driver] = df_month[driver]
    model_df[target] = df_month[target]

    # if we have to account for ice coverage
    if 'ICE'in drivers_list:
        model_df = model_df.loc[model_df['ICE'] >= ice_slider]
    
    # model_df['sst'] = df_month['sst']#.apply(lambda x:round(x,3))
    # model_df['dic'] = df_month['dic']#.apply(lambda x:round(x,3))
    # model_df['alk'] = df_month['alk']#.apply(lambda x:round(x,3))
    # model_df['fco2'] = df_month['fco2']#.apply(lambda x:round(x,3))
    # model_df['sosstsst'] = df_month['sosstsst']#.apply(lambda x:round(x,3))
    # model_df['DIC'] = df_month['DICP']#.apply(lambda x:round(x,3))
    # model_df['ALK'] = df_month['ALK']#.apply(lambda x:round(x,3))
    # model_df['fco2_pre'] = df_month['fco2_pre']#.apply(lambda x:round(x,3))
    return model_df

#---------------------------- 2. 2x2 Gridding. ----------------------------

def get_cell_range(start, end,cell_width =10):
    '''
    get the ranges of the cells
    '''
    num_iter = (abs(start) + abs(end))/cell_width
    range_lst = []
    for i in range(int(num_iter)+1):
        range_lst.append(start+i*cell_width)
#         print(start+i*cell_width)
    return range_lst

def build_grids(df_month,cell_width=2):
    # Prepare the cells
    nav_lat_grids = get_cell_range(start = -90, end = 90 ,cell_width = cell_width)
    nav_lon_grids = get_cell_range(start = -180, end = 180 ,cell_width = cell_width)
    
    if nav_lat_grids[-1] != 90:
        nav_lat_grids.append(90)
        
    if nav_lon_grids[-1] != 180:
        nav_lon_grids.append(180)
        
    # Build the grids. Store in a list.
    grids_df_lst=[]
    for lat_i in range(len(nav_lat_grids)):
        for lon_j in range(len(nav_lon_grids)):
            if((nav_lat_grids[lat_i] == 90) or (nav_lon_grids[lon_j] == 180)):
                break
            elif ((lat_i == len(nav_lat_grids) - 1) or (lon_j == len(nav_lon_grids) - 1)):
                break
            else:
                _df_ = df_month.loc[
                    (df_month['nav_lat'] >= nav_lat_grids[lat_i]) & 
                    (df_month['nav_lat'] <  nav_lat_grids[lat_i+1]) &
                    (df_month['nav_lon'] >= nav_lon_grids[lon_j]) & 
                    (df_month['nav_lon'] <  nav_lon_grids[lon_j+1])
                                ]
                grids_df_lst.append(_df_)
    
    # print(f"\n Total no. of generated cells: {len(grids_df_lst)}")
    
    return grids_df_lst


#---------------------------- 3. Multivariate Linear Regression. ----------------------------

def fit_multivariate_lin_regression(grids_df_lst, feat_names, target):
    '''
    https://satishgunjal.com/multivariate_lr_scikit/ 
    '''
    # Fit Regression model
    grid_reg_score = []
    grid_reg_coef = []
    grid_reg_intercept = []
        
    data_count = []
    
    count = 0
    
    for grid_i in grids_df_lst:
#         feat_lst=[]
#         for index, row in grid_i.iterrows():
#             row_feat_lst = [row[feat] for feat in feat_names]
#             feat_lst.append(np.array(row_feat_lst))
        
#         X = grid_i.values[:,2:5]
        # print(grid_i.columns) # --> for testing
        X = grid_i[feat_names].values
        y = np.array(grid_i[target].values)
   
        if np.isnan(X).any():
            print(X)
            raise ValueError
        
        if (len(X) == 0) or (len(y) == 0) or(len(X) == 1) or (len(y) == 1) :
            grid_reg_score.append(None)
            grid_reg_coef.append(None)
            grid_reg_intercept.append(None)
            data_count.append(len(X))
            
        else:
            data_count.append(len(X))
        
            # y = mX + c
            lin_reg = linear_model.LinearRegression().fit(X, y)
#             lin_reg = linear_model.Ridge(alpha=1.0).fit(X, y)

            # X2 = sm.add_constant(X)
            # est = sm.OLS(y, X2).fit()
            # _p_ = est.pvalues
            
            grid_reg_coef.append(lin_reg.coef_) #slope m
            grid_reg_intercept.append(lin_reg.intercept_) #intercept c
            grid_reg_score.append(lin_reg.score(X, y)) #quality or a confidence score
               
            count = count + 1
    
    
    df_column_names = ['cell_id','lon_min','lon_max','lat_min','lat_max', 'data_count',
                       'reg_score','reg_coef', 'reg_intercept','p_intercept']
    
        
    save_df = pd.DataFrame(columns=df_column_names)
    nav_lat_max_lst = []
    nav_lat_min_lst = []
    nav_lon_max_lst = []
    nav_lon_min_lst = []

    for grid_i in grids_df_lst:
        nav_lat_max_lst.append(grid_i['nav_lat'].max())
        nav_lat_min_lst.append(grid_i['nav_lat'].min())
        nav_lon_max_lst.append(grid_i['nav_lon'].max())
        nav_lon_min_lst.append(grid_i['nav_lon'].min())
    
    save_df['cell_id'] = range(0, len(grids_df_lst))
    save_df['lon_min'] = nav_lon_min_lst
    save_df['lon_max'] = nav_lon_max_lst
    save_df['lat_min'] = nav_lat_min_lst
    save_df['lat_max'] = nav_lat_max_lst
    save_df['data_count'] = data_count
    
    save_df['reg_score'] = grid_reg_score
    save_df['reg_coef'] = grid_reg_coef
    save_df['reg_intercept'] = grid_reg_intercept

    print("---> Finished.")
    return save_df

#---------------------------- 4. Plot Coefficients/ Slopes of Multivariate Linear Regression. ----------------------------

def plot_slope_maps(data_df, input_select_year, input_select_month, driver):
    buf = io.BytesIO() # in-memory files
    plt.figure(figsize=(12, 8))
    # fig = plt.figure(figsize=(20, 16), edgecolor='w')
    world_map = Basemap(projection='cyl', resolution='c',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, )

    # m.shadedrelief()
    ## Fill the land mass and lakes
    world_map.fillcontinents(color='black') #color_lake='aqua'

    # plt.title(f'Slope SST in {y} - {m}', fontsize=20)
    plt.title(f'Slope of {driver} in {input_select_month}, {input_select_year}', fontsize=20)

    if driver == 'SST':
        vmin = -10
        vmax = 30
    if driver == 'DICP':
        vmin = 0 
        vmax = 50
    if driver == 'DIC':
        vmin = 0 
        vmax = 50
    if driver == 'ALK':
        vmin = 0
        vmax = 50
    if driver == 'SAL':
        vmin = -5 
        vmax = 5
    # else:
    #     vmin = -5
    #     vmax = 5
    world_map_scatter =world_map.scatter(data_df['nav_lon'], data_df['nav_lat'],s = 5, c = data_df[driver],
                        vmin=vmin, vmax =vmax, cmap='RdYlBu_r', edgecolors='none')

    cbar = plt.colorbar(world_map_scatter, shrink = 0.5)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(f'Slope {driver}', fontsize=20)


    local_file_path = f'tmp/slopes_{driver}_{str(input_select_year)}_{input_select_month}.jpg'
    plt.savefig(local_file_path)
    plt.savefig(buf, format = "jpg") # save to the above file object
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.
    slope_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    slope_fig_path = f"data:image/jpg;base64,{slope_data}"
    return slope_fig_path



