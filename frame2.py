from function2 import *
import pymssql
import glob
import lightgbm as lgb
import warnings
from sklearn.model_selection import GridSearchCV
import os
import joblib
import shutil
warnings.filterwarnings("ignore")
'''训练模型'''

shutil.rmtree('model')
os.mkdir('model')

# ---数据部分---

time_unit = '月'   # 数据长度单位
unit = '月'   # 收益频率
n_train, n_test = 12, 12   # 训练集和测试集长度
n_evaluate = 12   # 验证集长度
response_type = '收益率'

file_names = sorted(glob.glob('dataset/factorPool_PV_20230303/因子中性化数据0707/*'))   # 加载文件名称

# ---模型部分---

# 模型参数设置
params = {'n_estimators': 200,
                'num_leaves': 64,
                'min_child_samples': 40,
                'min_child_weight': 0.001,
                'subsample': 1,
                'colsample_bytree': 1,
                'reg_alpha': 100,
                'reg_lambda': 100,
                'learning_rate': 0.05,
                'boosting_type': 'gbdt',
                # 'objective':custom_asymmetric_train
                'objective': 'regression',
                'verbosity': 2
                # 'first_metric_only':True,
                # 'force_col_wise':True
                }  # 默认参数

# 模型训练
model_time = '2020-01-01 00:00:00'   # 第一个模型训练时间
while valid_modle_date(file_names, model_time, n_train, n_test, n_evaluate, time_unit):
    print('训练' + model_time + '的模型')

    # 加载数据日历
    train_Calender = Load_Calender(file_names, str(pd.to_datetime(model_time) - DateOffset(weeks=n_train)) , n_train, time_unit, '日')
    test_Calender = Load_Calender(file_names, str(pd.to_datetime(model_time) - DateOffset(weeks=n_train + n_test)), n_test, time_unit, '日')
    # 加载数据文件
    train_files = filter_file_name(file_names, train_Calender)
    test_files = filter_file_name(file_names, test_Calender)
    # 加载训练集数据
    train_x, train_y = read_data(train_files, response_type, unit)
    # 加载测试集数据
    test_x, test_y = read_data(test_files, response_type, unit)

    # 训练模型
    gbm = lgb.LGBMRegressor(**params)

    # 网格搜索，参数优化
    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [20, 40]
    }
    lgbm = GridSearchCV(gbm, param_grid, scoring = 'neg_mean_squared_error')
    lgbm.fit(train_x.drop(['TRADE_DT', 'S_INFO_WINDCODE'], axis=1), train_y, eval_set=[(test_x.drop(['TRADE_DT', 'S_INFO_WINDCODE'], axis=1), test_y)], early_stopping_rounds=5)

    # 清内存
    del train_x, train_y, test_x, test_y

    # 模型存储
    joblib.dump(lgbm, 'model/{}.pkl'.format(model_time[:10]))

    print(model_time + '模型训练完成')

    model_time = str(pd.to_datetime(model_time) + DateOffset(months=12))   # 调整每个模型之间的间隔

    break


