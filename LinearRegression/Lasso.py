# 数据读取及基本处理
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV
from sklearn import metrics

# 读入数据
df = pd.read_csv("FE_day_T.csv")
#print(df.head())

# 数据分离
y = df['cnt']
X = df.drop('cnt', axis = 1)

lcv = linear_model.LassoCV(alphas=np.array([0.01, 0.1, 1, 10, 100]))

result = {}
k = []
def ridge_score(X,y):
    X = X.values
    y = y.values

    # 𝐾折交叉验证数据划分
    fold = KFold(5, shuffle=False)  # 切分5部分
    i = 0
    for train_index, test_index in fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lcv.fit(X_train, y_train)
        y_pred = lcv.predict(X_test)

        result[i] = lcv.coef_
        k.append(metrics.mean_squared_error(y_test, y_pred))
        print("RMSE:{0}".format(metrics.mean_squared_error(y_test, y_pred)))
        i = i + 1


ridge_score(X, y)
print(result[k.index(min(k))])
