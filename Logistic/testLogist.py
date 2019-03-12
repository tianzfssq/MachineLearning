'''
数据集共9个字段:
pregnants：怀孕次数
Plasma_glucose_concentration：口服葡萄糖耐量试验中2小时后的血浆葡萄糖浓度
blood_pressure：舒张压，单位:mm Hg
Triceps_skin_fold_thickness：三头肌皮褶厚度，单位：mm
serum_insulin：餐后血清胰岛素，单位:mm
BMI：体重指数（体重（公斤）/ 身高（米）^2）
Diabetes_pedigree_function：糖尿病家系作用
Age：年龄
Target：标签， 0表示不发病，1表示发病

二、作业要求：
用5折交叉验证，分别用log似然损失和正确率，对Logistic回归模型的正则超参数调优。（各50分）
'''

# 数据读取及基本处理
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# 读入数据
df = pd.read_csv("FE_pima-indians-diabetes-FE.csv")

# 数据分离
y = df['Target']
X = df.drop('Target', axis = 1)

# 稀疏数据-start
# 若数据集特征多为0,可将原始数据变为稀疏数据,可减少训练时间
# 查看一个学习期是否支持稀疏数据,可以看其fit函数是否支持:X{array-like, aparse matrix}
# 可自行使用timeit比较稠密数据和稀疏数据的训练时间
# from scipy.sparse import csr_matrix
# X_train = csr_matrix(X)
# 本数据即特征为0不算多,不使用稀疏
# 稀疏数据-end


feat_names = X.columns

lr = LogisticRegression()

# 交叉验证用于评估模型性能和进行参数调优(模型选择)
# 分类任务中交叉验证缺省是采用StratifiedKFold
# accuracy neg_log_loss
from sklearn.model_selection import cross_val_score
loss = cross_val_score(lr, X, y, cv=5, scoring='accuracy')
print('logloss of each flod is:', -loss)
print('cv logloss is:', -loss.mean())


# Logistic Regression + GridSearchCV
# logistic回归的需要调整超参数有:C(正则系数,一般在log域(取log后的值), 均匀设置候选参数)和正则函数penalty(L2/L1)
# 目标函数为J = C*sum(logloss(f(xi), yi)) + penalty
#　在sklearn框架下，不同学习期的参数调整步骤相同：
# 1. 设置参数搜索范围
# 2. 生成学习器示例（参数设置）
# 3. 生成GridSearchCV的示例(参数设置)
# 4. 调用GrdSearchCV的fit方法

X_train = X.values
y_train = y.values

penaltys = ['l1', 'l2']
Cs = [0.1, 1, 10, 100, 1000]
tuned_parammeters = dict(penalty=penaltys, C = Cs)

lr_penalty = LogisticRegression(solver='liblinear')

scorings = ['accuracy','neg_log_loss']

for s in scorings:
    grid = GridSearchCV(lr_penalty, tuned_parammeters, cv=5, scoring=s)
    grid.fit(X_train, y_train)

    print(s, ' score:', abs(grid.best_score_))
    print(s, ' params:', grid.best_params_)
    print('-------------------------------')





