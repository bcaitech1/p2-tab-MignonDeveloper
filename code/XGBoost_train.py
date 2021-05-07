# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import json, os
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb


from utils import seed_everything, get_config, print_score
from preprocess import feature_engineering

# 기본 프로젝트 디렉터리
PROJECT_PATH = "/opt/ml/online_retail_prediction"

# 데이터가 저장된 디렉터리
BASE_DATA_PATH = "/opt/ml/online_retail_prediction/input"


class CFG:
    objective = "binary"
    boosting_type = "gbdt"
    metric = "auc"
    feature_fraction = 0.8
    bagging_fraction = 0.8
    bagging_freq = 1
    n_estimators = 10000
    early_stopping_rounds = 100
    verbose = -1
    n_jobs = -1

    predict_year_month = '2011-12'
    seed = 42 # random seed
    total_thres = 300 # total threshold
    folds = 10 # number of k-fold
    model = 'LightGBM' # model
    description = 'Modeling' # description

    train_data_path = os.path.join(BASE_DATA_PATH, 'train.csv') # train csv 파일
    sample_submission_path = os.path.join(BASE_DATA_PATH, 'sample_submission.csv') # train csv 파일
    config_path = './config/config.json'
    docs_path = os.path.join(PROJECT_PATH, 'docs') # result, visualization 저장 경로
    model_path = os.path.join(PROJECT_PATH, 'models') # trained model 저장 경로


def get_data():
    # 데이터 파일 읽기
    data = pd.read_csv(CFG.train_data_path, parse_dates=['order_date'])
    return data


def get_from_dataset():
    train = pd.read_csv(os.path.join(CFG.docs_path,'dataset','train_399_best.csv'))
    test = pd.read_csv(os.path.join(CFG.docs_path,'dataset','test_399_best.csv'))
    features = train.drop(columns=['customer_id', 'label', 'year_month']).columns
    y = train['label']

    return train, test, y, features


def train_model(dtrain, dvalid, categorical_features='auto'):
    model_params = {
            'objective': 'binary:logistic', # 이진 분류
            'eval_metric': CFG.metric, # 평가 지표 설정
            'subsample': 0.9527699531356614, # 피처 샘플링 비율
            'colsample_bytree': 0.6666816689708133, # 데이터 샘플링 비율
            'n_estimators': CFG.n_estimators, # 트리 개수
            'early_stopping_rounds': CFG.early_stopping_rounds,
            'learning_rate': 0.06806593719106756,
            'seed': CFG.seed,
            'booster': 'gbtree',
            'verbosity': 0,
            'max_depth': 5,
        }
    
    model = xgb.train(
            params = model_params,
            dtrain = dtrain,
        )

    return model


def make_xgb_oof_prediction(train, y, test, features, categorical_features='auto', folds=10):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CFG.seed)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = xgb.DMatrix(x_tr, label=y_tr)
        dvalid = xgb.DMatrix(x_val, label=y_val)
        dtest = xgb.DMatrix(x_test)
        
        # LightGBM 모델 훈련
        model = train_model(dtrain, dvalid)

        # Validation 데이터 예측
        val_preds = model.predict(dvalid)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1}")
        print_score(y_val, val_preds)
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += model.predict(dtest) / folds
        
        # 폴드별 피처 중요도 저장
        # fi[f'fold_{fold+1}'] = model.feature_importances_

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_oof)

    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR = 1 - TNR')
    plt.ylabel('TPR = Recall')
    plt.savefig(os.path.join(CFG.docs_path, 'ROC_curce_oof.png'))
        
    # 폴드별 피처 중요도 평균값 계산해서 저장 
    # fi_cols = [col for col in fi.columns if 'fold_' in col]
    # fi['importance'] = fi[fi_cols].mean(axis=1)
    fi = []
    
    return y_oof, test_preds, fi


def inference(test_preds):
    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(CFG.sample_submission_path)
    
    # 테스트 예측 결과 저장
    sub['probability'] = test_preds
    
    # 제출 파일 쓰기
    sub.to_csv(os.path.join(CFG.docs_path ,'result', 'output_XGBoost.csv'), index=False)


def main():
    # config.json parsing
    get_config(CFG)

    # fix seed for reproducible model
    seed_everything(CFG.seed)

    # train.csv data 가져오기
    data = get_data()

    # 피처 엔지니어링 실행 및 Dataset 저장
    train_data, test_data = feature_engineering(data, CFG.predict_year_month, CFG.total_thres)
    train_data.to_csv(os.path.join(CFG.docs_path ,'dataset', 'train.csv'), index=False)
    test_data.to_csv(os.path.join(CFG.docs_path ,'dataset', 'test.csv'), index=False)
    
    # 저장된 Dataset 불러오기
    train, test, y, features = get_from_dataset()

    # Cross Validation Out Of Fold로 catboost 모델 훈련 및 예측
    y_oof, test_preds, fi = make_xgb_oof_prediction(train, y, test, features, folds=CFG.folds)

    # 테스트 결과 만들기
    inference(test_preds)


if __name__ == "__main__":
    main()