import xarray as xr
import pandas as pd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import axes
import colorcet as cc


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
from sklearn.preprocessing import StandardScaler
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

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 18 
mpl.rcParams['ytick.labelsize'] = 18 


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
    model_df['area'] = df_month['e1t'] * df_month['e2t']
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
                       'reg_score','reg_coef', 'reg_intercept',]
    
        
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
        vmax = 2
    if driver == 'DIC':
        vmin = 0 
        vmax = 2
    if driver == 'ALK':
        vmin = -2
        vmax = 5
    if driver == 'SAL':
        vmin = -5 
        vmax = 5
    else:
        vmin = -5
        vmax = 5
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

#---------------------------- 5. Run 1st round of HC and Plot Dendrograms. ----------------------------

def standardized_array(arr):
    scaler = StandardScaler()
    transformed_arr = scaler.fit_transform(arr)
    return transformed_arr

def get_standardized_array(df, col_name):
    mean_ = df[col_name].mean()
    std_ = df[col_name].std()
    return (df[col_name] - mean_) / std_

def run_hc(hc_df, drivers, file_name):

    
    norm_hc_df = hc_df    
    
    print("HC df is \n", hc_df)

    print("Norm HC df is\n", norm_hc_df)
    
    new_column_names = []

    for driver in drivers:
        lbl = f'slope_{driver}'
        new_lbl = f'slope_{driver}_std'

        # transformed_arr = standardized_array(arr = hc_df[lbl].values.reshape(-1,1)) 
        # norm_hc_df[new_lbl] = transformed_arr.flatten().tolist()

        # norm_hc_df[new_lbl] = get_standardized_array(df = hc_df, col_name = lbl)

        norm_hc_df[new_lbl] = (hc_df[lbl] - hc_df[lbl].mean())/hc_df[lbl].std()
        new_column_names.append(new_lbl)
    
    print("Standardizing driver slopes: \n", norm_hc_df.head(5))


    X = norm_hc_df[new_column_names].values
    Z = hierarchy.linkage(X, method='ward')


    buf = io.BytesIO() # in-memory files

    plt.figure(figsize=(20,12))
    plt.title(file_name,fontsize=30)
    plt.xlabel('Individual grid cells', fontsize=30)
    plt.ylabel('Ward Distance', fontsize=30)

    dendrogram_plot = hierarchy.dendrogram(Z, color_threshold=0)

    local_file_path = f'tmp/original_{file_name}.jpg'
    plt.savefig(local_file_path) # save locally
    plt.savefig(buf, format = "jpg") # save to the above file object (Runtime memory)
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.
    dendro_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    dendro_fig_path = f"data:image/jpg;base64,{dendro_data}"
    print('---> Returning the dendrogram path.')

    return norm_hc_df, Z, dendro_fig_path


#---------------------------- 5. Calculate BIC and no. of clusters for pair of delta_var and delta_dist. ----------------------------

def get_feat_arr(dendro_dict_lst, index_, arr):
    feat_stack = np.array([])
    for item in dendro_dict_lst:
            if item['id'] == index_:
                feat_stack = item['feat_stack']
                break
    if feat_stack.size == 0:
        feat_stack = [arr[index_]]
    return feat_stack

def get_grid_index_list(dendro_dict_lst, index_, df):
    _l_ = []
    for item in dendro_dict_lst:
            if item['id'] == index_:
                _l_ = item['grid_index']
                break
    if len(_l_) == 0:
        _l_ = [df.index[index_]]
    return _l_

def build_dendrogram_dict(Z, norm_hc_df, X):

    tot_children = len(norm_hc_df)
    clust_num = tot_children
    print(f'Start Cluster number = {clust_num}')
    dendro_dict_lst = []
    # index_count = len(Z) - 1
    index_count = 0
    for row in Z:
        dendro_dict ={}
        dendro_dict['id'] = clust_num    
        dendro_dict['info'] = row
        dendro_dict['feat_stack'] = np.array([])
        dendro_dict['variance'] = np.array([])
        dendro_dict['Z_index'] = index_count
        dendro_dict['grid_index'] = []

        dendro_dict_lst.append(dendro_dict)
        clust_num = clust_num + 1    
        index_count = index_count + 1
    print(f'End Cluster number = {clust_num}')  

    ## For connected leaf nodes
    count = 0
    for dendro in dendro_dict_lst:
        arr = dendro['info']
        first_index = int(arr[0])
        second_index = int(arr[1])
        child_num = arr[3]
        if child_num == 2:
            dendro['feat_stack'] =np.append([X[first_index]],[X[second_index]],axis = 0)
            dendro['variance'] = np.var(dendro['feat_stack'])
            _l_ = [norm_hc_df.index[first_index]]
            _l_.append(norm_hc_df.index[second_index])
            dendro['grid_index'] = _l_

            count = count + 1

    ## For connected leaf nodes and sub trees and more
    count = 0
    for dendro in dendro_dict_lst:
        arr_ = dendro['info']
        first_index = int(arr_[0])
        second_index = int(arr_[1])
        child_num = arr_[3]
        if child_num > 2:
            left_index = get_feat_arr(dendro_dict_lst = dendro_dict_lst, 
                            index_ = first_index, 
                            arr = X)
            
            right_index = get_feat_arr(dendro_dict_lst = dendro_dict_lst, 
                            index_ = second_index, 
                            arr = X)  
            dendro['feat_stack'] =np.append(left_index,right_index,axis = 0)
            dendro['variance'] = np.var(dendro['feat_stack'])
            
            left_grid_index = get_grid_index_list(dendro_dict_lst = dendro_dict_lst,
                                                index_ = first_index, df = norm_hc_df)
            right_grid_index = get_grid_index_list(dendro_dict_lst = dendro_dict_lst,
                                                index_ = second_index, df = norm_hc_df)
            # appending lists to one list
            dendro['grid_index'] = [*left_grid_index, *right_grid_index]
            count = count + 1

    return dendro_dict_lst


def run_through_dendrogram(delta_var, delta_dist, dendro_dict_lst):

    process_left = []
    process_right = []

    cluster_list = []
    z_index_list = []

    # get the top most element from the dendrogram and set is as "Start".
    start = dendro_dict_lst[-1]

    head = start['id']
    current_left = start['info'][0]
    current_right = start['info'][1]
    current_height = start['info'][2]
    current_variance = start['variance']
    current_z_index = start['Z_index']

    while(head):
        left_height = None
        left_variance = None
        left_z_index = None
        right_height = None
        right_variance = None
        right_z_index = None

        # left_child_num = 1
        # right_child_num = 1

        for item in dendro_dict_lst:
            if item['id'] == current_left:
                left_height = item['info'][2]
                left_variance = item['variance']
                left_z_index = item['Z_index']

        for item in dendro_dict_lst:
            if item['id'] == current_right:
                right_height = item['info'][2]
                right_variance = item['variance']
                right_z_index = item['Z_index']

        if right_height:
            change_height_right = abs(current_height - right_height)
            change_variance_right = abs(current_variance - right_variance)
            if ((change_height_right > delta_dist) and (change_variance_right > delta_var)):
                process_right.append(current_right)
            else:
                cluster_list.append(current_right)
                z_index_list.append(right_z_index)

        else:
            cluster_list.append(current_right)
            z_index_list.append(current_right)

        if left_height:
            change_height_left = abs(current_height - left_height)
            change_variance_left = abs(current_variance - left_variance)

            if ((change_height_left > delta_dist) and (change_variance_left > delta_var)): 
                #change_variance_left = np.inf
                    process_left.append(current_left)
            else:
                    # splitting the current head
                    cluster_list.append(current_left)
                    z_index_list.append(left_z_index)
        else:
            cluster_list.append(current_left)
            z_index_list.append(current_left)

        ## Reassign head
        if process_left:
            head = process_left.pop()
        elif process_right:
            head = process_right.pop()
        else:
            head = None

        ## Reassign
        if head:
            for item in dendro_dict_lst:
                if item['id'] == head:
                    current_left = item['info'][0]
                    current_right = item['info'][1]
                    current_height = item['info'][2]
                    current_variance = item['variance']
                    current_z_index = item['Z_index']
    return cluster_list,  z_index_list           


def get_all_clusters(dendro_dict_lst, cluster_list, norm_hc_df):
    heights_lst = []
    grid_index_list = []
    for c in cluster_list:
        found = False
        for item in dendro_dict_lst:
                if item['id'] == c:
    #                 print(item['id'])
                    found = True
                    ht = item['info'][2]
                    heights_lst.append(ht)
                    grid_index_list.append(item['grid_index'])
        if not found:
#             print(f"!!! Singleton cluster at {c}")
            heights_lst.append(0.0)
            grid_index_list.append([norm_hc_df.index[int(c)]])
    
    ## Extract clusters
    cluster_lbl = 1
    l__indices = []
    l__lbls = []
    for grid_list in grid_index_list:
        if type(grid_list) != list:
            l__indices.append(grid_list)
            l__lbls.append(cluster_lbl)
        else:
            for i in grid_list:
                # print(f'going wrong: ', i)
                l__indices.append(i)
                l__lbls.append(cluster_lbl)
        cluster_lbl = cluster_lbl + 1

    _df_ = pd.DataFrame(l__lbls, index =l__indices,columns =['cluster'])
    cluster_df = pd.merge(norm_hc_df, _df_, left_index=True, right_index=True)
    return cluster_df

def get_bic_aic_score(cluster_df, X):
    #size of data set -> N datapoints with d no. of features
    N, d = X.shape
    #unique labels
    labels_unique = cluster_df['cluster'].unique()

    loglikelihood = 0  
    for lbl in labels_unique:
        df_lbl = cluster_df.loc[cluster_df['cluster'] == lbl]
        loglikelihood =loglikelihood + np.log(len(df_lbl)/N)

    #BIC
    BIC = -2 * loglikelihood + d * np.log(N)
    AIC = -2 * loglikelihood + 2*d
    return BIC, AIC, labels_unique

def plot_bic_score_cluster_no(Z, norm_hc_df, year, month):

    df_column_names = norm_hc_df.columns
    std_columns = [column for column in df_column_names if "_std" in column]
    X = norm_hc_df[std_columns].values

    dendro_dict_lst = build_dendrogram_dict(Z, norm_hc_df, X)

    del_variance_list = np.arange(0.1, 1.0, 0.1).tolist()
    del_distance_list = np.arange(20, 40, 1).tolist()

    plot_del_var = []
    plot_del_dist = []
    bic_scores = []
    aic_scores = []
    tot_clusters = []

    for del_dist in del_distance_list:
        for del_var in del_variance_list:
    #         print(del_var, del_dist)
            cluster_list, z_index_list = run_through_dendrogram(del_var, del_dist, dendro_dict_lst)
            cluster_df = get_all_clusters(dendro_dict_lst, cluster_list, norm_hc_df)
            BIC, AIC, labels_unique = get_bic_aic_score(cluster_df, X)
            
            # Add to the lists
            plot_del_var.append(del_var)
            plot_del_dist.append(del_dist)
            bic_scores.append(BIC)
            aic_scores.append(AIC)
            tot_clusters.append(len(labels_unique))

    df = pd.DataFrame(list(zip(plot_del_dist, plot_del_var, bic_scores, aic_scores, tot_clusters )),
               columns =['Del_Dist (Change in Distance)', 'Del_Var (Change in Variance)', 'BIC Score', 'AIC Score', 'Total Clusters'])

    print(f"---> Building Cluster no. bubble graph for {month},{year}.")
    bic_cluster_num_fig = px.scatter(df, x="Del_Dist (Change in Distance)", y="Del_Var (Change in Variance)",
                                        size='Total Clusters',color= 'BIC Score', 
                                        hover_data=['Total Clusters', 'AIC Score'],)
    bic_cluster_num_fig.update_layout(title = f'BIC Score and total no. of clusters generated wrt change in variance and distance in {month},{year}',)

    return dendro_dict_lst, bic_cluster_num_fig

#---------------------------- 6. Cut dendrogram. ----------------------------

# https://stackoverflow.com/questions/73103010/matching-up-the-output-of-scipy-linkage-and-dendrogram

def append_index(n, i, cluster_id_list, Z):
    # refer to the recursive progress in
    # https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/cluster/hierarchy.py#L3549

    # i is the idx of cluster(counting in all 2 * n - 1 clusters)
    # so i-n is the idx in the "Z"
    if i < n:
        return
    aa = int(Z[i - n, 0])
    ab = int(Z[i - n, 1])

    append_index(n, aa, cluster_id_list, Z)
    append_index(n, ab, cluster_id_list, Z)

    cluster_id_list.append(i-n)
    # Imitate the progress in hierarchy.dendrogram
    # so how `i-n` is appended , is the same as how the element in 'icoord'&'dcoord' be.
    return

def get_linkid_clusterid_relation(Z):
    Zs = Z.shape
    n = Zs[0] + 1
    i = 2 * n - 2
    cluster_id_list = []
    append_index(n, i, cluster_id_list, Z)
    # cluster_id_list[i] is the cluster idx(in Z) that the R['icoord'][i]/R['dcoord'][i] corresponds to

    dict_linkid_2_clusterid = {linkid: clusterid for linkid, clusterid in enumerate(cluster_id_list)}
    dict_clusterid_2_linkid = {clusterid: linkid for linkid, clusterid in enumerate(cluster_id_list)}
    return dict_linkid_2_clusterid, dict_clusterid_2_linkid


def plot_detected_clusters_on_dendrogram(Z, z_index_list, heights_lst, delta_var, delta_dist):

    buf = io.BytesIO() # in-memory files
    plt.figure(figsize=(20, 12)).set_facecolor('white')
    plt.title(f"Dendrograms at delta_var = {round(delta_var,2)}, delta_dist = {round(delta_dist,2)}, #Clusters = {len(heights_lst)}",
            fontsize=20)

    dict_linkid_2_clusterid, dict_clusterid_2_linkid = get_linkid_clusterid_relation(Z)
    
    dend = hierarchy.dendrogram(Z, color_threshold=0)

    plt.xlabel("Individual grid cells", fontsize=20)
    plt.ylabel("Ward Distance", fontsize =20)
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = 18 
    mpl.rcParams['ytick.labelsize'] = 18

    LBL_COUNT = 1
    for z, h in zip(z_index_list, heights_lst):
        link_id = dict_clusterid_2_linkid[z]
        i = dend['icoord'][link_id]
        x = 0.5 * sum(i[1:3])
        y = h
        plt.plot(x, y, 'ro',c='red', markersize=5) #label = f'height = {y}'
        label = f"{LBL_COUNT}"
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the text
                    textcoords="offset points", # how to position the text
                    xytext=(10,10), # distance from text to points (x,y)
                    ha='left', # horizontal alignment can be left, right or center
                    color='black',
                    fontsize=16)
        plt.axhline(y=h, color='black', linestyle='--', linewidth = 0.5)  
    #     plt.axhline(y=h, color='black', linestyle='--', linewidth = 0.5, label = f'{LBL_COUNT} -> {h}')  
        
        LBL_COUNT = LBL_COUNT + 1
        
    # # for height in heights_lst:
    # #     plt.axhline(y=height, color='black', linestyle='--', label = f'{height}')  
    # # plt.legend()

    local_file_path = f'tmp/cut_dendrogram_{delta_dist}_{delta_var}.jpg'
    plt.savefig(local_file_path) # save locally
    plt.savefig(buf, format = "jpg") # save to the above file object (Runtime memory)
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.
    dendro_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    dendro_fig_path = f"data:image/jpg;base64,{dendro_data}"
    print('---> Returning the dendrogram path.')

    return dendro_fig_path

def detect_clusters_on_dendrogram(norm_hc_df, delta_var, delta_dist, dendro_dict_lst):

    df_column_names = norm_hc_df.columns
    std_columns = [column for column in df_column_names if "_std" in column]

    X = norm_hc_df[std_columns].values
    Z = hierarchy.linkage(X, method='ward')


    cluster_list, z_index_list = run_through_dendrogram(delta_var, delta_dist, dendro_dict_lst)

    heights_lst = []
    grid_index_list = []
    for c in cluster_list:
        found = False
        for item in dendro_dict_lst:
                if item['id'] == c:
    #                 print(item['id'])
                    found = True
                    ht = item['info'][2]
                    heights_lst.append(ht)
                    grid_index_list.append(item['grid_index'])
        if not found:
            print("singleton found")
            print(c)
            heights_lst.append(0.0)
            grid_index_list.append([norm_hc_df.index[int(c)]])

    fig_path = plot_detected_clusters_on_dendrogram(Z, z_index_list, heights_lst, delta_var, delta_dist)

    return grid_index_list, fig_path


#---------------------------- 7. Plot Clustered Maps. ----------------------------


def get_cluster_map(grid_index_list, norm_hc_df, grids_df_lst, drivers, delta_var, delta_dist, year, month ):
    
    buf = io.BytesIO() # in-memory files
    clrs = sns.color_palette(cc.glasbey, n_colors=len(grid_index_list)) # clrs = sns.color_palette("hls", num)

    cluster_lbl = 1
    l__indices = []
    l__lbls = []
    for grid_list in grid_index_list:
        if type(grid_list) != list:
            l__indices.append(grid_list)
            l__lbls.append(cluster_lbl)
        else:
            for i in grid_list:
                # print(f'going wrong: ', i)
                l__indices.append(i)
                l__lbls.append(cluster_lbl)
        cluster_lbl = cluster_lbl + 1

    _df_ = pd.DataFrame(l__lbls, index =l__indices,columns =['cluster'])
    hc_df = pd.merge(norm_hc_df, _df_, left_index=True, right_index=True)

    labels_all = hc_df['cluster'].values #all labels
    #size of data set -> N datapoints with d no. of features
    N = len(hc_df)
    d = len(drivers)
    #unique labels
    labels_unique = hc_df['cluster'].unique()

    loglikelihood = 0  
    for lbl in labels_unique:
        _df_ = hc_df.loc[hc_df['cluster'] == lbl]
        loglikelihood =loglikelihood + np.log(len(_df_)/N)

    #BIC
    BIC = -2 * loglikelihood + d * np.log(N)
    # AIC = -2 * loglikelihood + 2*d

    appended_data = []
    for index, row in hc_df.iterrows():
        # get the corresponding grid from the list of grids
        _df_ = grids_df_lst[index]
        _df_['grid_id'] = index
        _df_['cluster'] = row['cluster']
        for driver in drivers:
            lbl_name = f'slope_{driver}'
            _df_[lbl_name] = row[lbl_name]
        _df_['BIC'] = BIC
        _df_['delta_var'] = delta_var
        _df_['delta_dist'] = delta_dist
        appended_data.append(_df_)

    appended_data = pd.concat(appended_data)


    fig = plt.figure(figsize=(20, 16), edgecolor='w')
    plt.title(f'Carbon regimes of {year}, {month} with BIC Score = {BIC}, parameters = {delta_dist} and {delta_var}',fontsize=20)
    plt.xlabel('Longitude', fontsize=20, labelpad=40)
    plt.ylabel('Latitude', fontsize=20, labelpad=40)

    regime_map = Basemap(projection='cyl', resolution='c',llcrnrlat=-90, urcrnrlat=90,llcrnrlon=-180, urcrnrlon=180, )
    ## Draw the coast.
    # regime_map.drawcoastlines()

    ## Draw the land shades.
    # regime_map.shadedrelief()

    ## Fill the land mass and lakes
    regime_map.fillcontinents(color='black') #color_lake='aqua'

    ## draw parallels and meridians.
    # regime_map.drawparallels(np.arange(-90,91,10),labels=[1,1,0,1], fontsize=12)
    # regime_map.drawmeridians(np.arange(-180,181,10),labels=[1,1,0,1], rotation=45, fontsize=12)

    ##color the sea/oceans
    # regime_map.drawmapboundary(fill_color='aqua')

    count = 0
    arr = np.sort((appended_data['cluster'].unique()))
    for cluster_num in arr:
        _df_ = appended_data.loc[appended_data['cluster'] == cluster_num]
        # lbl = f"{int(cluster_num)} --> {len(_df_)}"
        lbl = f"{int(cluster_num)} ({round(_df_['area'].sum()/1000000,2)} km sq)"
        regime_map.scatter(_df_['nav_lon'], _df_['nav_lat'], latlon=True,marker='.', 
                color=clrs[count],label = lbl)
        count = count + 1


    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    lgnd = plt.legend(by_label.values(), by_label.keys(),loc='best', ncol=1, fontsize=20,bbox_to_anchor=(1.0, 0.8))
    for handle in lgnd.legendHandles:
        handle.set_linewidth(15)


    local_file_path = f'tmp/cluster_map_{delta_dist}_{delta_var}_{year}_{month}.jpg'
    plt.savefig(local_file_path) # save locally
    plt.savefig(buf, format = "jpg") # save to the above file object (Runtime memory)
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.
    map_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    map_fig_path = f"data:image/jpg;base64,{map_data}"
    print('---> Returning the map path.')

    return hc_df, clrs, appended_data, map_fig_path

#---------------------------- 8. Cluster Analysis ----------------------------

def get_scatter_plot_slopes(driver_pairs, hc_df, clrs):

    buf = io.BytesIO() # in-memory files

    if len(driver_pairs)%2 == 0:
        fig_rows = len(driver_pairs)/2
    else:
        fig_rows = (len(driver_pairs) + 1)/2

    fig, ax = plt.subplots(nrows=fig_rows, ncols=2)

    hc_df_rounded = hc_df.round(2)

    count = 0
    for row in ax:
        for col in row:
            driver1 = driver_pairs[count][0]
            driver2 = driver_pairs[count][1]
            tot_clusters = np.sort(hc_df_rounded['cluster'].unique())
            for t, c in zip(tot_clusters, clrs):
                df = hc_df_rounded.loc[hc_df_rounded['cluster'] == t]
                x = df[f'slope_{driver1}'].values
                y = df[f'slope_{driver2}'].values
                col.scatter(x, y, color=c, label = f'{t}')

                col.legend(loc='best', bbox_to_anchor=(1.15, 1),fancybox=True, shadow=True, ncol=1)
                col.set_title(f"Slope {driver1} vs. Slope {driver2}", fontsize=20)
                col.set_ylabel(f"Slope {driver1} ", fontsize=20)
                col.set_xlabel(f"Slope {driver2}", fontsize=20)
    
    local_file_path = f'tmp/scatter_plots.jpg'
    plt.savefig(local_file_path) # save locally
    plt.savefig(buf, format = "jpg") # save to the above file object (Runtime memory)
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.
    fig_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    fig_path = f"data:image/jpg;base64,{fig_data}"

    return fig_path

def analyse_clusters(drivers, hc_df, clrs):
    driver_pairs = []
    for i in range(len(drivers)):
        for j in range(i + 1, len(drivers)):
            driver_pairs.append((drivers[i], drivers[j]))
    
    ## Scatter plots
    fig_path = get_scatter_plot_slopes(driver_pairs, hc_df, clrs)

    ## Summary Table

    ## Random Forest
    
    # fig, ax = plt.subplots(2, 2)
    # for dp in driver_pairs:
    #     get_scatter_plot_slopes(dp[0], dp[1], hc_df, clrs)

    return fig_path
