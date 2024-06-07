import sys, os
import pickle
sys.path.append("..") 



def get_queries_dict(filename):
	#key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("Queries Loaded.")
		return queries

TRAIN_FILE = './oxford_evaluation_query_cut.pickle'
TRAINING_QUERIES= get_queries_dict(TRAIN_FILE)
print(TRAINING_QUERIES[0])
print(len(TRAINING_QUERIES))
