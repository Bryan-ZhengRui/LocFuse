from datasets.datasets_oxford import *
import models2
import os, sys, tqdm
import torch
from tools.utils import *
from tools.visualize import *
import time

def test_runtime():
    Dir = 'weights2/tmp_9_22_best/'
    weights = 'weight_best.pth'
    load_path = os.path.join(Dir,weights)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda:0")
    test_dataset = TEST_Robotcar_Dataset()
    database_dataset = Data_Robotcar_Dataset()
    net = models2.FusionNet()
    net.load_state_dict(torch.load(load_path))
    net.to(device)
    net.eval()
    recall = np.zeros(25)
    count = 0
    wrgbs = []
    wbevs = []
    similarity = []
    one_percent_recall = []
    query_features = []
    dataset_features = []
    nearst_dis_gt_sum = 0
    with open(test_query_path, 'rb') as file:
        queries = pickle.load(file)
    with open(test_database_path, 'rb') as file:
        database = pickle.load(file)    
    num_of_each_run_query = [len(queries[i]) for i in range(len(queries))]
    num_of_each_run_database = [len(database[i]) for i in range(len(database))]
    sum_num_of_each_run_query = [sum(num_of_each_run_query[:i]) for i in range(len(num_of_each_run_query))]  # [0, 100, 202, ...]
    sum_num_of_each_run_database = [sum(num_of_each_run_database[:i]) for i in range(len(num_of_each_run_database))]  # [0, 400, 802, ...]
    run_num = len(num_of_each_run_query) 
    begin_time = time.time()
    #-------------------Get query's embeddings------------------------
    test_query_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                                    pin_memory=True, drop_last = False, num_workers = 4)
    with torch.no_grad():
        for i, data in enumerate(test_query_loader, 0):
            rgb, bev = data['query']['rgb'].to(device), data['query']['bev'].to(device)
            batch_feature, _, _, wrgb, wbev = net(rgb, bev)
            batch_feature = batch_feature.detach().cpu().numpy()
            query_features.append(batch_feature)
            wrgb = wrgb.detach().cpu().numpy()
            wbev = wbev.detach().cpu().numpy()
            wrgbs.append(wrgb)
            wbevs.append(wbev)   
    end_time = time.time() 
    print('total rumtime:',end_time-begin_time)  
    print('each rumtime:',(end_time-begin_time)/sum(num_of_each_run_query))  
    query_features = np.vstack(query_features)
    #---------------------test_D_hybrid_time-----------------------------
    n,_ = query_features.shape
    array_384 = np.zeros((n, 512))
    array_384[:, :256] = query_features  
    array_384[:, 256:] = np.random.rand(n, 256)
    query_features = array_384
    #---------------------------------------------------------------------
    wrgbs = np.vstack(wrgbs)  
    wbevs = np.vstack(wbevs)  
    wrgb_mean = np.mean(wrgbs)
    wbev_mean = np.mean(wbevs)
    #-------------------Get database's embeddings----------------------
    test_database_loader = torch.utils.data.DataLoader(database_dataset, batch_size=1, shuffle=False, 
                                                       pin_memory=True, drop_last = False, num_workers = 4)
    with torch.no_grad():
        for i, data in enumerate(test_database_loader, 0):
            rgb, bev = data['query']['rgb'].to(device), data['query']['bev'].to(device)
            batch_feature, _, _, wrgb, wbev = net(rgb, bev)
            batch_feature = batch_feature.detach().cpu().numpy()
            dataset_features.append(batch_feature)
    dataset_features = np.vstack(dataset_features)
    
    #---------------------test_D_hybrid_time-----------------------------
    n2,_ = dataset_features.shape
    array_384 = np.zeros((n2, 512))
    array_384[:, :256] = dataset_features  
    array_384[:, 256:] = np.random.rand(n2, 256)
    dataset_features = array_384
    #---------------------------------------------------------------------
    
    num = 0
    begin_time2 = time.time()
    #-------------------evaluate for recall-------------------------
    for i in tqdm.tqdm(range(run_num), ncols=120):
        for j in range(run_num):
            if i == j:
                continue            
            st1 = sum_num_of_each_run_query[i]
            st2 = sum_num_of_each_run_database[j]
            end1 = st1 + num_of_each_run_query[i]
            end2 = st2 + num_of_each_run_database[j]
            query_data = query_features[st1:end1]
            database = dataset_features[st2:end2]
            num += len(query_data)
            pair_recall, pair_similarity, pair_opr, nearst_dis_gt = get_recall(j, i, database, query_data, queries)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)
            ave_recall = recall / count
            nearst_dis_gt_sum += nearst_dis_gt
    end_time2 = time.time()
    print('knn_search time:',(end_time2-begin_time2)/num)
    nearst_dis_gt_mean = nearst_dis_gt_sum/count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity, 'nearst_dis_gt_mean': nearst_dis_gt_mean, 
             'av_wrgb': wrgb_mean, 'av_wbev': wbev_mean}
    #---------------------get Recall@N plot----------------------------
    t = np.arange(1, 26, 1)
    y = stats['ave_recall']
    print(stats)
    plt.figure(figsize=(7,4),dpi = 300)
    plt.xlabel('num')
    plt.ylabel('Recall(%)')
    plt.title('Recall@N_Robotcar',fontsize=15)
    plt.plot(t,y,marker="o",color="blue")
    plt.savefig(os.path.join(Dir,"RecallN.pdf"), dpi=500)
    
    return stats



if __name__ == "__main__":
    
    test_runtime( )