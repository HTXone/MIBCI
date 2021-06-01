# 用xgb包能导出dump_model查看每棵树的形状
import xgboost
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from xgboost.sklearn import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import numpy as np
from xgboost import plot_importance

data = []
labels = []
with open('test.txt') as ifile:
    for line in ifile:
        tokens = line.strip().split('\t')
        data.append([float(tk) for tk in tokens[2:]])
        labels.append(tokens[1])
x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)
y[labels == '1'] = 1

# 拆分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

dtrain = xgboost.DMatrix(x_train, label=y_train)
dtest = xgboost.DMatrix(x_test)
params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'max_depth': 2,
          'lambda': 10,
          'subsample': 0.75,
          'colsample_bytree': 0.75,
          'min_child_weight': 1,
          'eta': 0.025,
          'seed': 0,
          'nthread': 8,
          'silent': 1}
watchlist = [(dtrain, 'train')]
bst = xgboost.train(params, dtrain, num_boost_round=100, evals=watchlist)
ypred = bst.predict(dtest)
y_pred = (ypred >= 0.5) * 1

###auc、混淆矩阵
from sklearn import metrics

print('AUC: %.4f' % metrics.roc_auc_score(y_test, ypred))
print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
print('Recall: %.4f' % metrics.recall_score(y_test, y_pred))
print('F1-score: %.4f' % metrics.f1_score(y_test, y_pred))
print('Precesion: %.4f' % metrics.precision_score(y_test, y_pred))
metrics.confusion_matrix(y_test, y_pred)

# 导出树结构
bst.dump_model('dump_model.txt')
ypred_contribs = bst.predict(dtest, pred_contribs=True)
xgboost.to_graphviz(bst, num_trees=1)  # 查看第n颗树

##重要性程度
import operator
import pandas as pd

importance = bst.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
print(importance)
df = pd.DataFrame(importance, columns=['feature', 'fscore']).sort_values(by='fscore', ascending=False)
# df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv('feature_importance.txt', index=False)

##计算KS
from scipy.stats import ks_2samp

get_ks = lambda y_pred, y_true: ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic
ks_test = get_ks(ypred, y_test)
