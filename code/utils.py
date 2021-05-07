import os
import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# config file parser
def get_config(CFG):
    with open(CFG.config_path) as f:
        config = json.load(f)
    
    CFG.objective = config["hyper_parameters"]["objective"]
    CFG.boosting_type = config["hyper_parameters"]["boosting_type"]
    CFG.metric = config["hyper_parameters"]["metric"]
    CFG.feature_fraction = config["hyper_parameters"]["feature_fraction"]
    CFG.bagging_fraction = config["hyper_parameters"]["bagging_fraction"]
    CFG.bagging_freq = config["hyper_parameters"]["bagging_freq"]
    CFG.n_estimators = config["hyper_parameters"]["n_estimators"]
    CFG.early_stopping_rounds = config["hyper_parameters"]["early_stopping_rounds"]
    CFG.verbose = config["hyper_parameters"]["verbose"]
    CFG.n_jobs = config["hyper_parameters"]["n_jobs"]

    CFG.predict_year_month = config["network_env"]["predict_year_month"]
    CFG.seed = config["network_env"]["seed"]
    CFG.total_thres = config["network_env"]["total_thres"]
    CFG.folds = config["network_env"]["folds"]
    # print(config)


# 시드 고정 함수
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# 평가 지표 출력 함수
def print_score(label, pred, prob_thres=0.5):
    tn, fp, fn, tp = confusion_matrix(label, pred > prob_thres).ravel()
    true_negative_rate = tn / (tn + fp)

    print('Precision: {:.5f}'.format(precision_score(label, pred>prob_thres)))
    print('Recall: {:.5f}'.format(recall_score(label, pred>prob_thres)))
    print('True Negative Rate: {:.5f}'.format(true_negative_rate))
    print('F1 Score: {:.5f}'.format(f1_score(label, pred>prob_thres)))
    print('ROC AUC Score: {:.5f}'.format(roc_auc_score(label, pred)))