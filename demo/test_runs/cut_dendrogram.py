import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib
matplotlib.use('Agg') ## Adding to avoid assertion failed error. It would shut down the server.
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import datetime

# for static images
import io
import base64

import plotly.express as px
import plotly.graph_objects as go


from build_cluster_map import extract_clusters


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

# https://stackoverflow.com/questions/73103010/matching-up-the-output-of-scipy-linkage-and-dendrogram

def append_index(n, i, cluster_id_list,Z):
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

def run_variance_height_method(delta_var, delta_dist,dendro_dict_lst):

    # delta_var = 0.2 # 0.0, 0.1
    # delta_dist = 20.0 # 1.0, 3.0

    process_left = []
    process_right = []

    cluster_list = []
    z_index_list = []


    start = dendro_dict_lst[-1]
    head = start['id']
    current_left = start['info'][0]
    current_right = start['info'][1]
    current_height = start['info'][2]
    current_variance = start['variance']
    current_z_index = start['Z_index']

    while(head):
        print()
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
            
            print(change_height_right)
            print(change_variance_right)

            if ((change_height_right > delta_dist) and (change_variance_right > delta_var)):
                process_right.append(current_right)
            else:
                cluster_list.append(current_right)
                z_index_list.append(right_z_index)

        else:
            cluster_list.append(current_right)
        

        if left_height:
            change_height_left = abs(current_height - left_height)
            change_variance_left = abs(current_variance - left_variance)
            
            print(change_height_left)
            print(change_variance_left)

            if ((change_height_left > delta_dist) and (change_variance_left > delta_var)): 
                #change_variance_left = np.inf
                    process_left.append(current_left)
            else:
                    # splitting the current head
                    cluster_list.append(current_left)
                    z_index_list.append(left_z_index)
        else:
            cluster_list.append(current_left)

    
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

    return cluster_list, z_index_list

def plot_clustered_dendrogram(Z, heights_lst, z_index_list, grid_index_list, delta_var, delta_dist, dict_clusterid_2_linkid,dend):
    ## Store Dendrogram
    buf = io.BytesIO() # in-memory files
    plt.figure(figsize=(20, 12)) #.patch.set_facecolor('red') <-- makes background red.
    # plt.title(f"Dendrograms with variance threshold = {delta_var}, distance threshold = {delta_dist}, #Clusters = {len(heights_lst)}. ",
    # The red dots represnt the dendrogram links of the clusters found. The dotted black lines highlights height at which clusters are found.",
            # fontsize=30)
    print(datetime.datetime.now())
    dend = hierarchy.dendrogram(Z, color_threshold=0)
    plt.xlabel('Individual grid cells', fontsize=30)
    plt.ylabel('Distance', fontsize=30)
    print(datetime.datetime.now())
    LBL_COUNT = 1 
    for z, h in zip(z_index_list, heights_lst):
        link_id = dict_clusterid_2_linkid[z]
        i = dend['icoord'][link_id]
        x = 0.5 * sum(i[1:3])
        y = h
        plt.plot(x, y, 's', markersize=8, c='red') #label = f'height = {y}' ## s=> Square
        label = f"{LBL_COUNT}"
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the text
                    textcoords="offset points", # how to position the text
                    xytext=(10,10), # distance from text to points (x,y)
                    ha='left', # horizontal alignment can be left, right or center
                    color='brown',
                    fontsize=40) 

        plt.axhline(y=h, color='black', linestyle='--',)  #label = f'{height}'
        LBL_COUNT = LBL_COUNT + 1

    # plt.legend()
    # plt.show()

    plt.savefig(f'tmp/dend_cluster_{str(datetime.datetime.now())}.png')
    plt.savefig(buf, format = "png") # save to the above file object
    plt.close() ## DO NOT COMMENT. It is to avoid assertion failed error. It would shut down the server.
    dendro_data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    dendro_fig_path = f"data:image/png;base64,{dendro_data}"
    print('---> Returning the clustered dendrogram path.')

    return dendro_fig_path



def build_cluster_points(delta_var, delta_dist, Z, norm_hc_df, X,dend):
    dict_linkid_2_clusterid, dict_clusterid_2_linkid = get_linkid_clusterid_relation(Z)
    dendro_dict_lst = build_dendrogram_dict(Z, norm_hc_df, X)

    if (len(dendro_dict_lst[-1]['grid_index']) == len(norm_hc_df)):
        cluster_list, z_index_list = run_variance_height_method(delta_var, delta_dist,dendro_dict_lst)
        heights_lst = []
        grid_index_list = []
        for c in cluster_list:
            found = False
            for item in dendro_dict_lst:
                    if item['id'] == c:
                        found = True
                        ht = item['info'][2]
                        heights_lst.append(ht)
                        grid_index_list.append(item['grid_index'])
            if not found:
                heights_lst.append(0.0)
                grid_index_list.append(c)

        dendro_fig_path = plot_clustered_dendrogram(Z, heights_lst, z_index_list, grid_index_list, delta_var, 
                                                delta_dist, dict_clusterid_2_linkid,dend)
        return dendro_fig_path, grid_index_list
    else:
        print("---> !!! Error in computation!")
        print(len(dendro_dict_lst[-1]['grid_index']))
        print(len(norm_hc_df))
        raise ValueError

def get_bic_scores(Z, norm_hc_df):
    X = norm_hc_df.values
    dendro_dict_lst = build_dendrogram_dict(Z, norm_hc_df, X)

    if (len(dendro_dict_lst[-1]['grid_index']) == len(norm_hc_df)):

        del_variance_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.7, 2.0]
        del_distance_list = [2, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]

        plot_var = []
        plot_dist = []
        plot_bic = []
        tot_cluster_num = []

        for del_dist in del_distance_list:
            for del_var in del_variance_list:
                cluster_list, z_index_list = run_variance_height_method(del_var, del_dist,dendro_dict_lst)
                heights_lst = []
                grid_index_list = []
                for c in cluster_list:
                    found = False
                    for item in dendro_dict_lst:
                            if item['id'] == c:
                                found = True
                                ht = item['info'][2]
                                heights_lst.append(ht)
                                grid_index_list.append(item['grid_index'])
                    if not found:
                        heights_lst.append(0.0)
                bic = extract_clusters(grid_index_list = grid_index_list, norm_hc_df = norm_hc_df, only_bic=True)

                plot_var.append(del_var)
                plot_dist.append(del_dist)
                plot_bic.append(bic)
                tot_cluster_num.append(len(heights_lst))

        return plot_var, plot_dist, plot_bic, tot_cluster_num
    else:
        print("---> !!! Error in BIC computation!")
        print(len(dendro_dict_lst[-1]['grid_index']))
        print(len(norm_hc_df))
        raise ValueError

def plot_bic_score_cluster_no(Z, norm_hc_df):
    var, dist, bic, cluster_num = get_bic_scores(Z, norm_hc_df)

    # print("---> Generating BIC heatmap.")
    # bic_dict = {'z': bic,
    #         'x': var,
    #         'y': dist}

    # bic_fig = go.Figure(data=go.Heatmap(bic_dict, reversescale=False, colorbar=dict(title='BIC Score')))
    # bic_fig.update_layout(title = 'BIC Score wrt change in variance and distance',
    #             legend_title_text='BIC Score', yaxis={"title": 'Change in Distance'},
    #               xaxis={"title": 'Change in Variance'}, showlegend=True)

    df = pd.DataFrame(list(zip(var, dist, bic, cluster_num)),
               columns =['Change in Variance', 'Change in Distance', 'BIC Score', 'Total Clusters'])
    # bic_fig = px.scatter(df, x="Change in Variance", y="Change in Distance",size='BIC Score', hover_data=['BIC Score'])

    print("---> Building Cluster no. bubble graph.")
    bic_cluster_num_fig = px.scatter(df, x="Change in Variance", y="Change in Distance",size='Total Clusters',
                                color= 'BIC Score', hover_data=['Total Clusters', 'BIC Score'],)
    bic_cluster_num_fig.update_layout(title = 'BIC Score and total no. of clusters generated wrt change in variance and distance',)
    
    return bic_cluster_num_fig