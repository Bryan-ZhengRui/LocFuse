import pandas as pd
import numpy as np
import os,sys
os.chdir(sys.path[0])
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "../"

runs_folder= "./RobotCar_samples/"
filename = "pointcloud_locations_train.csv"
rgb_fols="/rgb_cropped_train/"
bev_fols="/bev_cut_train/"

all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))

folders=[]

#All runs are used for training (both full and partial)
index_list=range(0,len(all_folders))
print("Number of runs: "+str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

#####For training and test data split#####
x_width=150
y_width=150
p1=[5735712.768124,620084.402381]
p2=[5735611.299219,620540.270327]
p3=[5735237.358209,620543.094379]
p4=[5734749.303802,619932.693364]   
# p=[p1,p2,p3,p4]
p=[p1,p2,p3,p4]


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set=False
    for point in points:
        if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
            in_test_set=True
            break
    return in_test_set
##########################################


def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing','easting']])
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=10, sort_results=True, return_distance=True)
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=50, sort_results=True, return_distance=True)
    ind_rr = tree.query_radius(df_centroids[['northing','easting']], r=400, sort_results=True, return_distance=True)
    queries={}
    for i in range(len(ind_nn[0])):
        query_rgb=df_centroids.iloc[i]["file_rgb"]
        query_bev=df_centroids.iloc[i]["file_bev"]
        if len(ind_nn[0][i])==1:
            positives=ind_nn[0][i].tolist()
        else:
            # positives=np.setdiff1d(ind_nn[i],[i]).tolist()
            positives = ind_nn[0][i].tolist()[1:]
        # negatives=np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
        ind_rr_0_0_list = ind_rr[0][i].tolist()
        ind_r_0_0_list = ind_r[0][i].tolist()
        negatives=np.setdiff1d(ind_rr_0_0_list, ind_r_0_0_list, assume_unique = True).tolist()
        # random.shuffle(negatives)
        queries[i]={"query_rgb":query_rgb,"query_bev":query_bev ,"positives":positives,"negatives":negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Done ", filename)


####Initialize pandas DataFrame
df_train= pd.DataFrame(columns=['file','northing','easting'])
df_test= pd.DataFrame(columns=['file','northing','easting'])

for folder in folders:
    df_locations= pd.read_csv(os.path.join(base_path,runs_folder,folder,filename),sep=',')
    df_locations['timestamp1']=runs_folder+folder+rgb_fols+df_locations['timestamp'].astype(str)+'.png'
    df_locations['timestamp2']=runs_folder+folder+bev_fols+df_locations['timestamp'].astype(str)+'_bev.png'
    df_locations=df_locations.rename(columns={'timestamp1':'file_rgb'})
    df_locations=df_locations.rename(columns={'timestamp2':'file_bev'})
    
    for index, row in df_locations.iterrows():
        if(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
            df_test=df_test.append(row, ignore_index=True)
        else:
            df_train=df_train.append(row, ignore_index=True)

print("Number of training submaps: "+str(len(df_train['file'])))
print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))
construct_query_dict(df_train,"training_queries_rob_cut.pickle")
construct_query_dict(df_test,"val_queries_rob_cut.pickle")

