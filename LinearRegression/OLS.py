# 数据读取及基本处理
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score,KFold,cross_validate
from sklearn import metrics

# 读入数据
df = pd.read_csv("FE_day_T.csv")
#print(df.head())

# 数据分离
y = df['cnt']
X = df.drop('cnt', axis = 1)

result = {}
k = []
# 最小二乘
lr = linear_model.LinearRegression()
def linear_score(X,y):
    X = X.values
    y = y.values

    # 𝐾折交叉验证数据划分
    fold = KFold(5, shuffle=False)  # 切分5部分


    i = 0
    for train_index, test_index in fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        result[i] = lr.coef_
        k.append(metrics.mean_squared_error(y_test, y_pred))


#        score = lr.score(X_test, y_test)
#        print(score)
#        print("MAE:{0}".format(metrics.mean_absolute_error(y_test, y_pred)))
        print("RMSE:{0}".format(k[i]))
        i = i+1


linear_score(X, y)

print(result[k.index(min(k))])

#lr = linear_model.LinearRegression()
#cross_result = cross_validate(lr, X, y, None, "neg_mean_squared_error", 5, return_train_score = False)
#print(cross_result.keys())
#print(cross_result['test_score'])