import numpy as np
from sklearn.neighbors import KDTree
import math

def get_recall(m, n, database_vectors, query_vectors, query_sets):
    
    database_output = database_vectors
    queries_output = query_vectors


    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    nearst_dis_gts = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        nearst_dis_gt = np.linalg.norm((queries_output[i] - database_output[true_neighbors[0]]), ord=2)
        nearst_dis_gts.append(nearst_dis_gt)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity) 
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    nearst_dis_gt = np.mean(nearst_dis_gts)
    return recall, top1_similarity_score, one_percent_recall, nearst_dis_gt

def get_recall_kitti(database_vectors, query_vectors, query_sets):
    
    database_output = database_vectors
    queries_output = query_vectors


    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors


    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[i+1800]    # {'query': path, 'northing': , 'easting': ,'index': []}
        true_neighbors = query_details['index']
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break


        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, one_percent_recall



def get_recall_kitti4overlap(database_vectors, query_vectors, query_sets):
    
    database_output = database_vectors
    queries_output = query_vectors


    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
   

    num_neighbors = 25
    recall = [0] * num_neighbors


    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(100, len(queries_output)):
        database_nbrs = KDTree(database_output[0:i-99])
        # i is query element ndx
        query_details = query_sets[i]    # {[],[],[],...}
        true_neighbors = query_details
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break


        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1
    
    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, one_percent_recall

def get_index_top1(m, n, database_vectors, query_vectors, query_sets):
    
    database_output = database_vectors
    queries_output = query_vectors

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    nearst_dis_gts = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)
    top1_indices = []
    top1_judge = []
    query_indices_n = []
    query_indices_e = []
    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        query_indices_n.append(query_details['northing'])
        query_indices_e.append(query_details['easting'])
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
        if indices[0][0] in true_neighbors:
            top1_indices.append(indices[0][0])
            top1_judge.append(1)
        else:
            top1_indices.append(indices[0][0])
            top1_judge.append(0)
            
    return top1_indices, top1_judge, query_indices_n, query_indices_e