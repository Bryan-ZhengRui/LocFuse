from datasets.datasets_oxford import *
from models2 import * 
import os, sys, tqdm
import torch
from tools.utils import *      
from tools.visualize import *      
from torch.utils.tensorboard import SummaryWriter 
from loss.loss_func import *  



config_filename = os.path.join('configs/config.yaml')
config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
test_query_path = config['Robotcar']['test_query_pickle_path']
# weights_saved_path = config['Experiments']['weights_saved_path']
Dir = "weights2/tmp_train_new/"
batchsize = 16

def evaluate(test_dataset, database_dataset, net, epoch, device):
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
    #-------------------Get query's embeddings------------------------
    test_query_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False, 
                                                    pin_memory=True, drop_last = False, num_workers = 8)
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
    query_features = np.vstack(query_features)
    wrgbs = np.vstack(wrgbs)  
    wbevs = np.vstack(wbevs)  
    wrgb_mean = np.mean(wrgbs)
    wbev_mean = np.mean(wbevs)
    #-------------------Get database's embeddings----------------------
    test_database_loader = torch.utils.data.DataLoader(database_dataset, batch_size=batchsize, shuffle=False, 
                                                       pin_memory=True, drop_last = False, num_workers = 8)
    with torch.no_grad():
        for i, data in enumerate(test_database_loader, 0):
            rgb, bev = data['query']['rgb'].to(device), data['query']['bev'].to(device)
            batch_feature, _, _, wrgb, wbev = net(rgb, bev)
            batch_feature = batch_feature.detach().cpu().numpy()
            dataset_features.append(batch_feature)
    dataset_features = np.vstack(dataset_features)
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
            
            pair_recall, pair_similarity, pair_opr, nearst_dis_gt = get_recall(j, i, database, query_data, queries)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)
            ave_recall = recall / count
            nearst_dis_gt_sum += nearst_dis_gt
    nearst_dis_gt_mean = nearst_dis_gt_sum/count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity, 'nearst_dis_gt_mean': nearst_dis_gt_mean, 
             'av_wrgb': wrgb_mean, 'av_wbev': wbev_mean}
    
    return stats


def main():
    #select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
    device = torch.device("cuda:0")
    writer = SummaryWriter(os.path.join(Dir,'log'))
    
    # weights = 'weight_best.pth'
    # load_path = os.path.join(Dir,weights)
    
    train_dataset = Robotcar_Dataset_qua(transform = True)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batchsize, shuffle=True, 
                                               pin_memory=True, drop_last=True, num_workers = 16)
    triple_net = FusionNet()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        triple_net = nn.DataParallel(triple_net)
        # triple_net.module.load_state_dict(torch.load(load_path))
        
    triple_net = triple_net.to(device)
    initial_lr = 1e-4
    epoch_num = 150
    optimizer = torch.optim.Adam(triple_net.parameters(), lr=initial_lr, weight_decay = 5e-5)
    optimizer = nn.DataParallel(optimizer).module #MutiGPU Training
    top1_recall = 0 
    for epoch in range(epoch_num):
        triple_net.train()
        if epoch > 0 and epoch <= 10:
            for p in optimizer.param_groups:
                p['lr'] -= initial_lr / 50
        elif epoch>10 and epoch % 2 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.91
        lr_rate = optimizer.state_dict()['param_groups'][0]['lr']      
        running_loss = 0.0
        running_loss_rgb = 0.0
        running_loss_bev = 0.0
        running_loss_fus = 0.0
        train_loader = tqdm.tqdm(train_loader, file=sys.stdout, ncols = 130)
        for i, data in enumerate(train_loader, 0):
            anchor_rgb = data['anchor']['rgb']
            anchor_bev = data['anchor']['bev']
            positive_rgb = data['positives']['rgb']
            positive_bev = data['positives']['bev']
            negative_rgb = data['negatives']['rgb']
            negative_bev = data['negatives']['bev']
            negative_rgb_n = data['negatives_nearst']['rgb']
            negative_bev_n = data['negatives_nearst']['bev']
            optimizer.zero_grad()
            anchor, a_rgb, a_bev, _, _ = triple_net(anchor_rgb.to(device), anchor_bev.to(device))
            positive, p_rgb, p_bev, _, _ = triple_net(positive_rgb.to(device), positive_bev.to(device)) 
            negative, n_rgb, n_bev, _, _ = triple_net(negative_rgb.to(device), negative_bev.to(device))
            negative_n, n_rgb_n, n_bev_n, _, _ = triple_net(negative_rgb_n.to(device), negative_bev_n.to(device))
            if epoch < 10:
                loss, loss_rgb, loss_bev, loss_fus, loss_class_dis = quadruplet_loss(anchor, a_rgb, a_bev, positive, p_rgb, p_bev, 
                                                                  negative, n_rgb, n_bev, negative_n, n_rgb_n, n_bev_n, 
                                                                  margin1 = 0.37, margin2 = 0.25, device=device)
            elif epoch < 30:
                loss, loss_rgb, loss_bev, loss_fus, loss_class_dis = quadruplet_loss(anchor, a_rgb, a_bev, positive, p_rgb, p_bev, 
                                                                  negative, n_rgb, n_bev, negative_n, n_rgb_n, n_bev_n, 
                                                                  margin1 = 0.39, margin2 = 0.27, device=device)
            elif epoch < 60:
                loss, loss_rgb, loss_bev, loss_fus, loss_class_dis = quadruplet_loss(anchor, a_rgb, a_bev, positive, p_rgb, p_bev, 
                                                                  negative, n_rgb, n_bev, negative_n, n_rgb_n, n_bev_n, 
                                                                  margin1 = 0.4, margin2 = 0.28, device=device)
            elif epoch < 100:
                loss, loss_rgb, loss_bev, loss_fus, loss_class_dis = quadruplet_loss(anchor, a_rgb, a_bev, positive, p_rgb, p_bev, 
                                                                  negative, n_rgb, n_bev, negative_n, n_rgb_n, n_bev_n, 
                                                                  margin1 = 0.41, margin2 = 0.29, device=device)
            else:
                loss, loss_rgb, loss_bev, loss_fus, loss_class_dis = quadruplet_loss(anchor, a_rgb, a_bev, positive, p_rgb, p_bev, 
                                                                  negative, n_rgb, n_bev, negative_n, n_rgb_n, n_bev_n, 
                                                                  margin1 = 0.42, margin2 = 0.3, device=device)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_rgb += loss_rgb.item()
            running_loss_bev += loss_bev.item()
            running_loss_fus += loss_fus.item()
            if i % 10 == 9:
                train_loader.desc = "[epoch {}/{}] lr:{}, loss: {:.5f}".format(epoch+1, epoch_num, 
                                                                               lr_rate, running_loss/10)
                writer.add_scalars('Train_loss', {'train_loss':running_loss/10}, epoch * len(train_loader) + i)
                writer.add_scalars('loss_isolated', {'bev_loss':running_loss_bev/10, 'rgb_loss':running_loss_rgb/10,
                                                    'fusion_loss':running_loss_fus/10, 'loss_class_dis': loss_class_dis}, epoch * len(train_loader) + i)
                running_loss = 0.0   
                running_loss_rgb = 0.0
                running_loss_bev = 0.0
                running_loss_fus = 0.0           

        val_rst = evaluate(net=triple_net, epoch=epoch, device=device, test_dataset = TEST_Robotcar_Dataset(transform=None), database_dataset = Data_Robotcar_Dataset())
        t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}  Avg_Nearst. distance: {:.4f} Avg. recall @N:'
        print(t.format(val_rst['ave_one_percent_recall'], val_rst['average_similarity'], val_rst['nearst_dis_gt_mean']))
        print(val_rst['ave_recall'])
        writer.add_scalars('Avg_Nearst_distance',  {'Nearst_distance': val_rst['nearst_dis_gt_mean']}, epoch)
        writer.add_scalars('Avg_weight',  {'wrgb_mean': val_rst['av_wrgb'], 'wbev_mean': val_rst['av_wbev']}, epoch)
        writer.add_scalars('Recall',  {'recall@1':val_rst['ave_recall'][0], 'recall 1%':val_rst['ave_one_percent_recall']}, epoch)
        
        if val_rst['ave_recall'][0] >= top1_recall:
            top1_recall = val_rst['ave_recall'][0]
            torch.save(triple_net.module.state_dict(), os.path.join(Dir,'weight_best.pth'))
        if epoch%5 == 0:
            torch.save(triple_net.module.state_dict(), os.path.join(Dir,'weight_epoch'+str(epoch+1)+'.pth'))
    writer.close()






if __name__ == "__main__":
    main()