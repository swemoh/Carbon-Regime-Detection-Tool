import colorcet as cc
import seaborn as sns
import pandas as pd
import numpy as np
import datetime

import matplotlib
matplotlib.use('Agg') ## Adding to avoid assertion failed error. It would shut down the server.
import matplotlib.pyplot as plt

# For plotting maps
import os
os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
from mpl_toolkits.basemap import Basemap
import plotly.express as px
import geopandas
# import plotly.graph_objs as go

# for static images
import io
import base64

#----------------------------------------------------------------------------------------------------

def get_colors(grid_index_list):
    num = len(grid_index_list)
    ## n random colors
    clrs = sns.color_palette(cc.glasbey, n_colors=num) 
    # clrs = sns.color_palette("hls", num)
    return clrs

def calculate_BIC(hc_df, X):
    labels_all = hc_df['cluster'].values #all labels
    #size of data set -> N datapoints with d no. of features
    N, d = X.shape
    #unique labels
    labels_unique = hc_df['cluster'].unique()

    loglikelihood = 0  
    for lbl in labels_unique:
        _df_ = hc_df.loc[hc_df['cluster'] == lbl]
        loglikelihood =loglikelihood + np.log(len(_df_)/N)

    #BIC
    BIC = -2 * loglikelihood + d * np.log(N)
    return BIC


# def plot_map_plotly(appended_data):
#     df = appended_data

#     import plotly.express as px

#     arr = np.sort((appended_data['cluster'].unique()))

#     sst_mean = []
#     dic_mean = []
#     alk_mean = []
#     fco2_mean = []
#     slope_sst_mean = []
#     slope_dic_mean = []
#     slope_alk_mean = []
#     cluster_number = []

    # for cluster_num in arr:
    #     _df_ = appended_data.loc[appended_data['cluster'] == cluster_num]
    #     sst_mean.append(_df_['sst'].mean())
    #     dic_mean.append(_df_['dic'].mean())
    #     alk_mean.append(_df_['alk'].mean())
    #     fco2_mean.append(_df_['fco2'].mean())
    #     slope_sst_mean.append(_df_['slope_sst'].mean())
    #     slope_dic_mean.append(_df_['slope_dic'].mean())
    #     slope_alk_mean.append(_df_['slope_alk'].mean())
    #     cluster_number.append(cluster_num)


    # fig = px.scatter_mapbox(appended_data, lat="nav_lat", lon="nav_lon", hover_name="cluster", color='cluster', zoom = 8,) #height=300
    # fig.update_layout(mapbox_style="open-street-map")
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # cluster_info_df = pd.DataFrame(list(zip(cluster_number, sst_mean, dic_mean, alk_mean, fco2_mean, 
    #                 slope_sst_mean, slope_dic_mean, slope_alk_mean )),
    #            columns =['Cluster Label', 'SST', 'DIC', 'ALK', 'fco2', 'SST Slope', 'DIC Slope', 'ALK Slope'])

    # print('---> Returning the clustered map path.')
    # return fig


def plot_map(appended_data, clrs):
    buf = io.BytesIO() # in-memory files
    # fig = plt.figure(figsize=(12, 8), edgecolor='w')
    fig = plt.figure(edgecolor='w', figsize=(12, 8))
    m = Basemap(projection='cyl', resolution='c',
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, )
    ## Draw the coast.
    # m.drawcoastlines()

    ## Draw the land shades.
    # m.shadedrelief()

    ## Fill the land mass and lakes
    m.fillcontinents(color='black') #color_lake='aqua'

    ## draw parallels and meridians.
    # m.drawparallels(np.arange(-90,91,10),labels=[1,1,0,1], fontsize=12)
    # m.drawmeridians(np.arange(-180,181,10),labels=[1,1,0,1], rotation=45, fontsize=12)

    ##color the sea/oceans
    # m.drawmapboundary(fill_color='aqua')

    count = 0
    arr = np.sort((appended_data['cluster'].unique()))
    # print(arr)

    # sst_mean = []
    # dic_mean = []
    # alk_mean = []
    # fco2_mean = []
    slope_sst_mean = []
    slope_dic_mean = []
    slope_alk_mean = []
    cluster_number = []

    for cluster_num in arr:
        _df_ = appended_data.loc[appended_data['cluster'] == cluster_num]
        _df_['marker_size'] = 0.1 #0.05
        _df_['marker_size'] = _df_['marker_size'].astype(np.float)

        # sst_mean.append(_df_['sst'].mean())
        # dic_mean.append(_df_['dic'].mean())
        # alk_mean.append(_df_['alk'].mean())
        # fco2_mean.append(_df_['fco2'].mean())
        slope_sst_mean.append(_df_['slope_sst'].mean())
        slope_dic_mean.append(_df_['slope_dic'].mean())
        slope_alk_mean.append(_df_['slope_alk'].mean())
        cluster_number.append(cluster_num)


        lbl = str(int(cluster_num))
        # lbl = f"{cluster_num} --> {len(_df_)}"
        m.scatter(_df_['nav_lon'], _df_['nav_lat'], latlon=True, marker='.', s= _df_['marker_size'],
                color=clrs[count], alpha = 1.0,
                label = lbl)
        count = count + 1

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    lgnd = plt.legend(by_label.values(), by_label.keys(),loc='upper left', ncol=10, title = 'Cluster labels',
                        fontsize=10, bbox_to_anchor=(0,-0.1)) #best, 1,1
    for handle in lgnd.legendHandles:
        handle.set_linewidth(15)
        handle.set_sizes([1.0])

    plt.title(f'Corresponding Clusters on the map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # plt.show()
    plt.savefig(f'tmp/cluster_map_{str(datetime.datetime.now())}.png')
    plt.savefig(buf, format = "png") # save to the above file object
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.

    map_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    map_fig_path = f"data:image/png;base64,{map_data}"


    cluster_info_df = pd.DataFrame(list(zip(cluster_number,slope_sst_mean, slope_dic_mean, slope_alk_mean )),
               columns =['Cluster Label','SST Slope', 'DIC Slope', 'ALK Slope'])

    print('---> Returning the clustered map path.')
    return map_fig_path ,cluster_info_df

def get_clustered_data_distribution(appended_data, clrs):
    buf = io.BytesIO() # in-memory files
    # fig = plt.figure(figsize=(20, 16), edgecolor='w')
    # with sns.plotting_context(rc={"axes.labelsize":28}):
    #     fig_dist =  
    
    appended_data = appended_data.astype({'cluster':int})
    sns.pairplot(data = appended_data,
                kind = "scatter",
            #     diag_kind="hist",
                hue= 'cluster',
                vars=["slope_sst", "slope_dic", "slope_alk"],
                palette = clrs,
            #     plot_kws=dict(marker="+", linewidth=1, alpha=0.4),    
                plot_kws=dict(alpha=0.5),
                diag_kws=dict(fill=False),
                corner=False,
                aspect=20/16)
    # plt.title(f'Linear regression coefficients distribution.')
    # plt.show()
    plt.savefig(f'tmp/cluster_dist_{str(datetime.datetime.now())}.png')
    plt.savefig(buf, format = "png") # save to the above file object
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.

    dist_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    data_dist_fig_path = f"data:image/png;base64,{dist_data}"
    return data_dist_fig_path



def extract_clusters(grids_df_lst=None, grid_index_list=None, norm_hc_df=None, only_bic=False):

    clrs = get_colors(grid_index_list)

    X = norm_hc_df.values

    cluster_lbl = 1
    l__indices = []
    l__lbls = []
    for grid_list in grid_index_list:
        for i in grid_list:
            l__indices.append(i)
            l__lbls.append(cluster_lbl)
        cluster_lbl = cluster_lbl + 1
    
    _df_ = pd.DataFrame(l__lbls, index =l__indices,columns =['cluster'])
    hc_df = pd.merge(norm_hc_df, _df_, left_index=True, right_index=True)

    bic = calculate_BIC(hc_df, X)

    if only_bic:
        return bic

    appended_data = []
    for index, row in hc_df.iterrows():
        # get the corresponding grid from the list of grids
        _df_ = grids_df_lst[index]
        _df_['cluster'] = row['cluster']
        _df_['slope_sst'] = row['slope_sst']
        _df_['slope_dic'] = row['slope_dic']
        _df_['slope_alk'] = row['slope_alk']
        appended_data.append(_df_)
    appended_data = pd.concat(appended_data)

    map_fig_path, cluster_info_df = plot_map(appended_data, clrs)
    # plotly_map = plot_map_plotly(appended_data)
    data_dist_fig_path = get_clustered_data_distribution(appended_data, clrs)
    return map_fig_path, data_dist_fig_path, cluster_info_df, appended_data, bic