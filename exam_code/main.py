"""Trains/evaluates NRNNMF models."""
from __future__ import absolute_import, print_function
# Standard modules
import argparse, json, time, os
# Third party modules
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
# Package modules
from nrnnmf.models import NRNNMF
    

def load_data(data_filename, drugMat_filename, geneMat_filename):
    data = pd.read_table(data_filename, index_col=0)
    drugMat = pd.read_table(drugMat_filename, index_col=0)
    geneMat = pd.read_table(geneMat_filename, index_col = 0)
    return data, drugMat, geneMat

def train(model, data_filename, drugMat_filename, geneMat_filename, sess, batch_size, max_iters, use_early_stop, early_stop_max_iter):
    aupr_list, auc_list = [], []
    train_aupr_list, train_auc_list = [], []
    #Load data
    data, drugMat, geneMat = load_data(data_filename, drugMat_filename, geneMat_filename)
    num_users = len(drugMat)
    num_items = len(geneMat)

    # data transformation
    interaction_list = []
    i = 0
    j = 0
    for drug in data.columns:
        for gene in data.index:
            interaction_list.append([i, j, data[drug][gene]])
            j += 1
        j = 0
        i += 1
    
    #ten fold cross validation
    interaction_list = pd.DataFrame(interaction_list, columns=['drug','gene', 'interaction'])
    # Shuffle
    interaction_list = (interaction_list.iloc[np.random.permutation(len(interaction_list))]).reset_index(drop=True)
    #ten fold cross validation
    kfold = StratifiedKFold(n_splits = 10)
    for num_pair in range(5):
        for train_ind, test_ind in kfold.split(interaction_list.ix[:, 0:2], interaction_list.ix[:, 2]):
            train_data = interaction_list.ix[train_ind]
            test_data = interaction_list.ix[test_ind]
            # Divide the train data to training dataset and validation dataset
            train_X, valid_X, train_y, valid_y = train_test_split(train_data.ix[:, 0:2], train_data.ix[:, 2], test_size = 0.02)
            train_data = pd.concat([train_X, train_y], axis = 1)
            valid_data = pd.concat([valid_X, valid_y], axis = 1)

            #initialize the model 
            model = NRNNMF(num_users, num_items, drugMat, geneMat, **model_params)
            model.init_sess(sess)

            #optimization
            model = optimization(model, train_data, valid_data, drugMat, geneMat, sess, batch_size, max_iters, use_early_stop, early_stop_max_iter)

            #record the performance of model
            train_scores = model.predict(train_data)
            train_prec, train_rec, train_thr = precision_recall_curve(train_data.ix[:, 2], np.array(train_scores))
            train_aupr_val = auc(train_rec, train_prec)
            train_fpr, train_tpr, train_thr = roc_curve(train_data.ix[:, 2], np.array(train_scores))
            train_auc_val = auc(train_fpr, train_tpr)
            train_aupr_list.append(train_aupr_val)
            train_auc_list.append(train_auc_val)        
            scores = model.predict(test_data, train_data)
            prec, rec, thr = precision_recall_curve(test_data.ix[:, 2], np.array(scores))
            aupr_val = auc(rec, prec)
            fpr, tpr, thr = roc_curve(test_data.ix[:, 2], np.array(scores))
            auc_val = auc(fpr, tpr)
            aupr_list.append(aupr_val)
            auc_list.append(auc_val)
            print('train_auc: {:3f}, train_aupr: {:3f}, auc: {:3f}, aupr: {:3f}'.format(train_auc_val, train_aupr_val, auc_val, aupr_val))
    return np.array(train_aupr_list, dtype=np.float64), np.array(train_auc_list, dtype=np.float64), np.array(aupr_list, dtype=np.float64), np.array(auc_list, dtype=np.float64)      
     

def optimization(model, train_data, valid_data, drugMat, geneMat, sess, batch_size, max_iters, use_early_stop, early_stop_max_iter):
    last_log = float("Inf")
        
    for t in range(max_iters):
        prev_valid_rmse = float("Inf")
        early_stop_iters = 0
        # Run SGD
        batch = train_data.sample(batch_size) if batch_size else train_data

        # Evaluate
        train_error = model.eval_loss(batch)
        train_rmse = model.eval_rmse(batch)
        valid_rmse = model.eval_rmse(valid_data)
        print("{:3f} {:3f}, {:3f}".format(train_error, train_rmse, valid_rmse))

        # Early stopping
        if use_early_stop:
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                model.train_iteration(batch)
            elif early_stop_iters == early_stop_max_iter:
                print("Early stopping ({} vs. {})...".format(prev_valid_rmse, valid_rmse))
                break
            elif valid_rmse > prev_valid_rmse:
                early_stop_iters += 1
        else:
            model.train_iteration(batch)
    return model

def mean_confidence_interval(data, confidence=0.95):
    
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, h


if __name__ == '__main__':
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains/evaluates NRNNMF models.')
    parser.add_argument('--adjacency', metavar='ADJACENCY_FILE', type=str, default='data/adjacency_matrix.txt',
                        help='the location of the adjacency matrix file')
    parser.add_argument('--drug', metavar='DRUG_SIMILARITY_FILE', type=str, default='data/drug_similarity.txt',
                        help='the location of the drug similarity file')
    parser.add_argument('--gene', metavar='GENE_SIMILARITY_FILE', type=str, default='data/gene_similarity.txt',
                        help='the location of the gene similarity file')
    parser.add_argument('--model-params', metavar='MODEL_PARAMS_JSON', type=str, default='{}',
                        help='JSON string containing model params')
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=10000,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--no-early', default=False, action='store_true',
                        help='disable early stopping')
    parser.add_argument('--early-stop-max-iter', metavar='EARLY_STOP_MAX_ITER', type=int, default=40,
                        help='the maximum number of iterations to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-iters', metavar='MAX_ITERS', type=int, default=1000,
                        help='the maximum number of iterations to allow the model to train for')

    # Parse args
    args = parser.parse_args()
    # Global args
    data_filename = args.adjacency
    drugMat_filename = args.drug
    geneMat_filename = args.gene
    model_params = json.loads(args.model_params)
    batch_size = args.batch
    use_early_stop = not(args.no_early)
    early_stop_max_iter = args.early_stop_max_iter
    max_iters = args.max_iters
    sess = tf.Session()
    tic = time.clock()
    train_aupr_vec, train_auc_vec, aupr_vec, auc_vec = train(model_params, data_filename, drugMat_filename, geneMat_filename, sess, batch_size, max_iters, use_early_stop, early_stop_max_iter)
    train_aupr_avg, train_aupr_conf = mean_confidence_interval(train_aupr_vec)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    train_auc_avg, train_auc_conf = mean_confidence_interval(train_auc_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    print ("Train: auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (train_auc_avg, train_aupr_avg, train_auc_conf, train_aupr_conf, time.clock()-tic))
    print ("Test: auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
