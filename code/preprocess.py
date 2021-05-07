import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta

# Machine learning
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres, print_log=False):
    df = df.copy()

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month'] < year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month'] == year_month]
    # print(len(cust))
    # print(df)

    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)
    # print(label)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label


def feature_preprocessing(train, test, features, do_imputing="zero", do_scaling="min_max"):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            # print(1)    
            cate_cols.append(f)
            
            x_tr[f] = x_tr[f].astype('category')
            x_te[f] = x_te[f].astype('category')


    if do_imputing == "median" or do_imputing == "mean":
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy=do_imputing)
        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    elif do_imputing == "zero":
        # 0으로 결측치 채우기
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])


    if do_scaling == "min_max":
        # min max scaling
        scaler = MinMaxScaler(feature_range=(-1,1))
        x_tr[features] = scaler.fit_transform(x_tr[features])
        x_te[features] = scaler.fit_transform(x_te[features])

    elif do_scaling == "standard":
        # min max scaling
        scaler = StandardScaler()
        x_tr[features] = scaler.fit_transform(x_tr[features])
        x_te[features] = scaler.fit_transform(x_te[features])

    elif do_scaling == "robust":
        # min max scaling
        scaler = RobustScaler()
        x_tr[features] = scaler.fit_transform(x_tr[features])
        x_te[features] = scaler.fit_transform(x_te[features])

    elif do_scaling == "quantile":
        # min max scaling
        scaler = QuantileTransformer()
        x_tr[features] = scaler.fit_transform(x_tr[features])
        x_te[features] = scaler.fit_transform(x_te[features])
    

    return x_tr, x_te, cate_cols


def mean_ewm(x):
    result = x.ewm(com=9.5).mean().mean()
    return result


def last_ewm(x):
    result = x.ewm(com=9.5).mean().iloc[-1]
    return result


def avg_diff(x):
    result = (x - x.shift(1)).mean()
    return result


def add_time_general_statistics(train, test, prev_ym_d, d):
    train = train.copy()
    test = test.copy()

    train_window_ym = []
    test_window_ym = []
    for month_back in [1, 2, 3, 5, 7, 12, 20, 23]:
        train_window_ym.append((prev_ym_d - dateutil.relativedelta.relativedelta(months=month_back)).strftime('%Y-%m'))
        test_window_ym.append((d - dateutil.relativedelta.relativedelta(months=month_back)).strftime('%Y-%m'))
    # print(train_window_ym)

    # aggregation 함수 선언
    agg_func = ['max','min','sum','mean','count','std','skew',mean_ewm,last_ewm,avg_diff]

    # group by aggregation with Dictionary
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
    }

    # general statistics for train data with time series
    for i, tr_ym in enumerate(train_window_ym):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['year_month'] >= tr_ym].groupby(['customer_id']).agg(agg_dict)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in train_agg.columns:
            new_cols.append(f'{level1}-{level2}-{i}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        if i == 0:
            train_data = train_agg
        else:
            train_data = train_data.merge(train_agg, on=['customer_id'], how='right')


    # general statistics for test data with time series
    for i, tr_ym in enumerate(test_window_ym):
        # group by aggretation 함수로 test 데이터 피처 생성
        test_agg = test.loc[test['year_month'] >= tr_ym].groupby(['customer_id']).agg(agg_dict)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in test_agg.columns:
            new_cols.append(f'{level1}-{level2}-{i}')

        test_agg.columns = new_cols
        test_agg.reset_index(inplace = True)
        
        if i == 0:
            test_data = test_agg
        else:
            test_data = test_data.merge(test_agg, on=['customer_id'], how='right')

    return train_data, test_data


def add_seasonality(train, test, prev_ym_d, d):
    train_window_ym = []
    test_window_ym = []
    for month_back in [1, 12]:
        train_window_ym.append(
            (
                (prev_ym_d - dateutil.relativedelta.relativedelta(months=month_back)).strftime('%Y-%m'),
                (prev_ym_d - dateutil.relativedelta.relativedelta(months=month_back+2)).strftime('%Y-%m')
            )
        )
        test_window_ym.append(
            (
                (d - dateutil.relativedelta.relativedelta(months=month_back)).strftime('%Y-%m'),
                (d - dateutil.relativedelta.relativedelta(months=month_back+2)).strftime('%Y-%m')
            )
        )
    
    # aggregation 함수 선언
    agg_func = ['max','min','sum','mean','count','std','skew',mean_ewm,last_ewm,avg_diff]

    # group by aggregation with Dictionary
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
    }

    # seasonality for train data with time series
    for i, (tr_ym, tr_ym_3) in enumerate(train_window_ym):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[(train['year_month'] >= tr_ym_3) & (train['year_month'] <= tr_ym)].groupby(['customer_id']).agg(agg_dict)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in train_agg.columns:
            new_cols.append(f'{level1}-{level2}-season{i}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        if i == 0:
            train_data = train_agg
        else:
            train_data = train_data.merge(train_agg, on=['customer_id'], how='right')


    # seasonality for test data with time series
    for i, (tr_ym, tr_ym_3) in enumerate(test_window_ym):
        # group by aggretation 함수로 train 데이터 피처 생성
        test_agg = test.loc[(test['year_month'] >= tr_ym_3) & (test['year_month'] <= tr_ym)].groupby(['customer_id']).agg(agg_dict)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for level1, level2 in test_agg.columns:
            new_cols.append(f'{level1}-{level2}-season{i}')

        test_agg.columns = new_cols
        test_agg.reset_index(inplace = True)
        
        if i == 0:
            test_data = test_agg
        else:
            test_data = test_data.merge(test_agg, on=['customer_id'], how='right')
    
    return train_data, test_data


def add_cumsum(train, test, prev_ym_d, d):
    train = train.copy()
    test = test.copy()

    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    train['cumsum_total_by_cust_id'] = train.groupby(['customer_id'])['total'].cumsum()
    train['cumsum_quantity_by_cust_id'] = train.groupby(['customer_id'])['quantity'].cumsum()
    train['cumsum_price_by_cust_id'] = train.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    train['cumsum_total_by_prod_id'] = train.groupby(['product_id'])['total'].cumsum()
    train['cumsum_quantity_by_prod_id'] = train.groupby(['product_id'])['quantity'].cumsum()
    train['cumsum_price_by_prod_id'] = train.groupby(['product_id'])['price'].cumsum()
    
    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    train['cumsum_total_by_order_id'] = train.groupby(['order_id'])['total'].cumsum()
    train['cumsum_quantity_by_order_id'] = train.groupby(['order_id'])['quantity'].cumsum()
    train['cumsum_price_by_order_id'] = train.groupby(['order_id'])['price'].cumsum()   

    #-----------------------------------------------------------------------------
    
    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    test['cumsum_total_by_cust_id'] = test.groupby(['customer_id'])['total'].cumsum()
    test['cumsum_quantity_by_cust_id'] = test.groupby(['customer_id'])['quantity'].cumsum()
    test['cumsum_price_by_cust_id'] = test.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    test['cumsum_total_by_prod_id'] = test.groupby(['product_id'])['total'].cumsum()
    test['cumsum_quantity_by_prod_id'] = test.groupby(['product_id'])['quantity'].cumsum()
    test['cumsum_price_by_prod_id'] = test.groupby(['product_id'])['price'].cumsum()
    
    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    test['cumsum_total_by_order_id'] = test.groupby(['order_id'])['total'].cumsum()
    test['cumsum_quantity_by_order_id'] = test.groupby(['order_id'])['quantity'].cumsum()
    test['cumsum_price_by_order_id'] = test.groupby(['order_id'])['price'].cumsum() 

    # agg_func = ['mean','max','min','sum','count','std','skew']
    agg_func = ['max','min','sum','mean','count','std','skew',mean_ewm,last_ewm,avg_diff]

    agg_dict = {
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
        'cumsum_total_by_prod_id': agg_func,
        'cumsum_quantity_by_prod_id': agg_func,
        'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
    }

    # group by aggretation 함수로 train 데이터 피처 생성
    train_agg = train.loc[train['year_month'] < prev_ym_d.strftime('%Y-%m')].groupby(['customer_id']).agg(agg_dict)

    # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
    new_cols = []
    for level1, level2 in train_agg.columns:
        new_cols.append(f'{level1}-{level2}')

    train_agg.columns = new_cols
    train_agg.reset_index(inplace = True)
    train_data = train_agg


    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.loc[test['year_month'] < d.strftime('%Y-%m')].groupby(['customer_id']).agg(agg_dict)

    # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
    new_cols = []
    for level1, level2 in test_agg.columns:
        new_cols.append(f'{level1}-{level2}')

    test_agg.columns = new_cols
    test_agg.reset_index(inplace = True)
    test_data = test_agg

    return train_data, test_data


def add_ts_diff(train, test, prev_ym_d, d):
    train = train.copy()
    test = test.copy()

    train['order_ts'] = train['order_date'].astype(np.int64) // 1e9
    train['order_ts_diff'] = train.groupby(['customer_id'])['order_ts'].diff()
    train['quantity_diff'] = train.groupby(['customer_id'])['quantity'].diff()
    train['price_diff'] = train.groupby(['customer_id'])['price'].diff()
    train['total_diff'] = train.groupby(['customer_id'])['total'].diff()

    test['order_ts'] = test['order_date'].astype(np.int64) // 1e9
    test['order_ts_diff'] = test.groupby(['customer_id'])['order_ts'].diff()
    test['quantity_diff'] = test.groupby(['customer_id'])['quantity'].diff()
    test['price_diff'] = test.groupby(['customer_id'])['price'].diff()
    test['total_diff'] = test.groupby(['customer_id'])['total'].diff()

    # agg_func = ['mean','max','min','sum','count','std','skew']
    agg_func = ['max','min','sum','mean','count','std','skew',mean_ewm,last_ewm,avg_diff]
    
    agg_dict = {
        'order_ts': ['first', 'last'],
        'order_ts_diff': agg_func,
        'quantity_diff': agg_func,
        'price_diff': agg_func,
        'total_diff': agg_func,
    }

    # group by aggretation 함수로 train 데이터 피처 생성
    train_agg = train.loc[train['year_month'] < prev_ym_d.strftime('%Y-%m')].groupby(['customer_id']).agg(agg_dict)

    # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
    new_cols = []
    for level1, level2 in train_agg.columns:
        new_cols.append(f'{level1}-{level2}')

    train_agg.columns = new_cols
    train_agg.reset_index(inplace = True)
    train_data = train_agg


    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.loc[test['year_month'] < d.strftime('%Y-%m')].groupby(['customer_id']).agg(agg_dict)

    # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
    new_cols = []
    for level1, level2 in test_agg.columns:
        new_cols.append(f'{level1}-{level2}')

    test_agg.columns = new_cols
    test_agg.reset_index(inplace = True)
    test_data = test_agg

    return train_data, test_data


def count_over_300(train, test):
    train = train.copy()
    test = test.copy()

    train_new_feature = train.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    train_new_feature = train_new_feature.groupby(['customer_id']).agg(lambda x: (x>300).sum())
    train_new_feature.columns = ['total_over_300']

    test_new_feature = test.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    test_new_feature = test_new_feature.groupby(['customer_id']).agg(lambda x: (x>300).sum())
    test_new_feature.columns = ['total_over_300']
    return train_new_feature, test_new_feature


def count_using_shopping(train, test):
    train = train.copy()
    test = test.copy()

    train_new_feature = train.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    train_new_feature = train_new_feature.groupby(['customer_id']).agg(lambda x: len(x))
    train_new_feature = pd.DataFrame(train_new_feature['year_month'])
    train_new_feature.columns = ['count_using_shopping']

    test_new_feature = test.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    test_new_feature = test_new_feature.groupby(['customer_id']).agg(lambda x: len(x))
    test_new_feature = pd.DataFrame(test_new_feature['year_month'])
    test_new_feature.columns = ['count_using_shopping']
    return train_new_feature, test_new_feature


def count_over_300_ratio(train, test):
    train = train.copy()
    test = test.copy()

    train_new_feature = train.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    train_new_feature = train_new_feature.groupby(['customer_id']).agg(lambda x: (x>300).sum() / len(x))
    train_new_feature.columns = ['total_over_300_ratio']

    test_new_feature = test.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    test_new_feature = test_new_feature.groupby(['customer_id']).agg(lambda x: (x>300).sum() / len(x))
    test_new_feature.columns = ['total_over_300_ratio']
    return train_new_feature, test_new_feature


def total_sum_per_count_using_shopping(train, test):
    train = train.copy()
    test = test.copy()

    train_new_feature = train.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    train_total_sum = train_new_feature.groupby(['customer_id']).agg('sum')
    train_total_count = train_new_feature.groupby(['customer_id']).agg(lambda x: len(x))
    train_new_feature = pd.DataFrame(train_total_sum['total'] / train_total_count['year_month'])
    train_new_feature.columns = ['total_sum_per_count_using_shopping']

    test_new_feature = test.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    test_total_sum = test_new_feature.groupby(['customer_id']).agg('sum')
    test_total_count = test_new_feature.groupby(['customer_id']).agg(lambda x: len(x))
    test_new_feature = pd.DataFrame(test_total_sum['total'] / test_total_count['year_month'])
    test_new_feature.columns = ['total_sum_per_count_using_shopping']
    return train_new_feature, test_new_feature


def devide_sum(x):
    total_average = x.sum() / len(x)

    answer = ''
    if total_average<0: answer = 'minus'
    elif total_average<20: answer = 'very low'
    elif total_average<40: answer = 'low'
    elif total_average<70: answer = 'middle-low'
    elif total_average<100: answer = 'middle'
    elif total_average<150: answer = 'middle-high'
    elif total_average<200: answer = 'almost'
    elif total_average<250: answer = 'high'
    elif total_average<270: answer = 'higgh'
    elif total_average<300: answer = 'hiigh'
    elif total_average<325: answer = 'veeery high'
    elif total_average<350: answer = 'very high'
    elif total_average<500: answer = 'very hiigh'
    elif total_average<1000: answer = 'super high'
    else: answer = 'epic high'

    return answer


def total_sum_per_count_using_shopping_cat(train, test):
    train = train.copy()
    test = test.copy()

    train_new_feature = train.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    train_new_feature = train_new_feature.groupby(['customer_id']).agg(devide_sum)
    train_new_feature.columns = ['total_sum_per_count_using_shopping_cat']

    test_new_feature = test.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    test_new_feature = test_new_feature.groupby(['customer_id']).agg(devide_sum)
    test_new_feature.columns = ['total_sum_per_count_using_shopping_cat']

    return train_new_feature, test_new_feature


def last_month_shopping(train,test, prev_ym_d, d):
    train = train.copy()
    test = test.copy()

    train_new_feature = train.groupby(['customer_id'], as_index=False)[['year_month']].max()
    train_new_feature = train_new_feature.groupby(['customer_id']).agg(
        lambda x: 24 - dateutil.relativedelta.relativedelta(
            prev_ym_d, datetime.datetime.strptime(max(x), "%Y-%m")
        ).months
    )
    train_new_feature.columns = ['last_month_shopping']

    test_new_feature = test.groupby(['customer_id'], as_index=False)[['year_month']].max()
    test_new_feature = test_new_feature.groupby(['customer_id']).agg(
        lambda x: 24 - dateutil.relativedelta.relativedelta(
            prev_ym_d, datetime.datetime.strptime(max(x), "%Y-%m")
        ).months
    )
    test_new_feature.columns = ['last_month_shopping']

    return train_new_feature, test_new_feature



def feature_engineering(df, year_month, total_thres):
    df = df.copy() 

    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym_d = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym_d.strftime('%Y-%m')

    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    
    # train, test 데이터 선택
    train = df[df['year_month'] < prev_ym]
    test = df[df['year_month'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_data = generate_label(df, prev_ym, total_thres)[['customer_id','year_month','label']]
    test_data = generate_label(df, year_month, total_thres)[['customer_id','year_month','label']]

    data_part_df = []
    data_part_df.append(add_time_general_statistics(train, test, prev_ym_d, d))
    data_part_df.append(add_seasonality(train, test, prev_ym_d, d))
    data_part_df.append(add_cumsum(train, test, prev_ym_d, d))
    data_part_df.append(add_ts_diff(train, test, prev_ym_d, d))
    data_part_df.append(count_over_300(train, test))
    data_part_df.append(count_using_shopping(train,test))
    data_part_df.append(total_sum_per_count_using_shopping(train,test))
    # data_part_df.append(total_sum_per_count_using_shopping_cat(train,test))
    data_part_df.append(count_over_300_ratio(train,test))
    # data_part_df.append(last_month_shopping(train,test, prev_ym_d, d))

    for train_data_part, test_data_part in data_part_df:
        train_data = train_data.merge(train_data_part, on=['customer_id'], how='left')
        test_data = test_data.merge(test_data_part, on=['customer_id'], how='left')

    features = train_data.drop(columns=['customer_id', 'label', 'year_month']).columns

    print(f"train_shape: {train_data.shape}")
    print(f"test_shape : {test_data.shape}")

    # train, test 데이터 전처리
    x_tr, x_te, cate_cols = feature_preprocessing(train_data, test_data, features, do_imputing="zero", do_scaling=None)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, cate_cols
