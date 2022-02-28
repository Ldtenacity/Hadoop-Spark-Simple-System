#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
#author: Liu Dong
#email: LdTenacity666@163.com
#corporation: Beijing Institute of Technology
#accomplished time:2021.9.8-2021.9.20


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("movies2.csv",encoding='gbk')


# In[4]:


df['category']=df['类别']
for i in range(0,250):
    df['category'].loc[i]=df['类别'][i].split()[0]
    #print(df['类别'][i].split()[0])
    #print(df['category'].loc[i])
print(df['category'])


# In[5]:


print(df['拍摄地'])
df['location']=df['拍摄地']
for i in range(0,250):
    df['location'].loc[i]=df['拍摄地'][i].split()[0]
    #print(df['类别'][i].split()[0])
    #print(df['category'].loc[i])
print(df['location'])


# In[6]:


print(df['年份'])
df['released-year']=df['年份']
for i in range(0,250):
    ls=df['年份'][i].split(',')
    #print(df['年份'][i].split(','))
    s=df['年份'][i].split(',')[0]
    if(len(ls)>1):
        str2=df['年份'][i].split(',')[1]
        s+=str2
    df['released-year'].loc[i]=s
print(df['released-year'])
print(min(df['released-year']))
print(df.iloc[52])


# In[7]:


sns.distplot(df['released-year'], hist=True, kde=True,
             bins=int(180), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.title('number of movies in douban TOP 250 per released-year')
plt.show()


# In[8]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data1 = np.array(df['评分'])
data2 = np.array(df['评价数'])
data3 = np.array(df['rank'])


def do_linearRegression(data,name):
    X = np.linspace(0, 199, 200).reshape(-1, 1)
    y = data[0:200] 
    X_future = np.linspace(200, 250, 50).reshape(-1, 1)
    y_future = data[200:250]
    model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    model.fit(X, y)

    predicted = model.predict(X)
    pred = model.predict(X_future)

    mse = np.sum(np.square(abs(y - predicted))) / 1000

    plt.figure()
    plt.title(name)
    plt.plot(X, y, c='b')
    plt.plot(X, predicted, c='r')
    plt.plot(X_future, pred, c='green')
    plt.plot(X_future, y_future, c='purple')
    plt.xlabel("rank")
    plt.ylabel("comments number")
    plt.show()


# In[9]:


do_linearRegression(data1,'linear regression model of score versus rank')


# In[10]:


do_linearRegression(data2,'linear regression model of comments number versus rank')


# In[11]:


#decision tree function
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
tdata1 = [np.array(df['评分']),"score"]
tdata2 = [np.array(df['评价数']),"number of comments"]
def decision_tree_regression(data):
    X = np.linspace(0, 99, 100).reshape(-1, 1)
    y = data[0][0:100]

    regr_1 = DecisionTreeRegressor(max_depth=5)
    regr_2 = DecisionTreeRegressor(max_depth=7)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    X_test = np.arange(0.0, 100.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black",c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=4", linewidth=2)
    plt.xlabel("rank")
    plt.ylabel(data[1])
    plt.title("Decision Tree Regression of "+data[1]+" versus rank")
    plt.legend()
    plt.show()


# In[12]:


decision_tree_regression(tdata1)


# In[13]:


decision_tree_regression(tdata2)


# In[14]:


#Boosted Decision Tree Function: Decision Tree Regression with AdaBoost

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

def boosted_decsion_tree_regression(datax,datay,target,fromm):
    rng = np.random.RandomState(1)
    X=datax.reshape(-1,1)
    y=datay

    regr_1 = DecisionTreeRegressor(max_depth=2)

    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2),
                              n_estimators=50, random_state=rng)

    regr_1.fit(X, y)
    regr_2.fit(X, y)

    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)

    plt.figure()
    plt.scatter(X, y, c="k", label="training samples")
    plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
    plt.plot(X, y_2, c="r", label="n_estimators=50", linewidth=2)
    plt.xlabel(fromm)
    plt.ylabel(target)
    plt.title("Boosted Decision Tree Regression of "+target+" versus "+fromm)
    plt.legend()
    plt.show()


# In[15]:


boosted_decsion_tree_regression(data3,data1,"score","rank")


# In[16]:


boosted_decsion_tree_regression(data3,data2,"number of comments","rank")


# In[17]:


boosted_decsion_tree_regression(data2,data1,"score","number of comments")


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
def PCR_PLS(data1,data2,name1,name2,company_name):
    X_train = data1[0:200].reshape(-1, 1)
    y_train = data2[0:200] 
    X_test = data1[201:250].reshape(-1, 1)
    y_test = data2[201:250]
    pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
    pcr.fit(X_train, y_train)
    pca = pcr.named_steps['pca']  

    pls = PLSRegression(n_components=1)
    pls.fit(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].scatter(pca.transform(X_test), y_test, alpha=.3, label='ground truth')
    axes[0].scatter(pca.transform(X_test), pcr.predict(X_test), alpha=.3,label='predictions')
    axes[0].set(xlabel=name1,ylabel=name2, title=company_name+' PCR')
    axes[0].legend()
    axes[1].scatter(pls.transform(X_test), y_test, alpha=.3, label='ground truth')
    axes[1].scatter(pls.transform(X_test), pls.predict(X_test), alpha=.3,label='predictions')
    axes[1].set(xlabel=name1,ylabel=name2, title=company_name+' PLS')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


# In[19]:


PCR_PLS(data1,data2,"score","number of comments","douban TOP 250")


# In[20]:


ax=sns.heatmap(df.corr())


# In[21]:


from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

np.random.seed(42)

n_samples, n_features = 50, 100
X = np.random.randn(n_samples, n_features)

idx = np.arange(n_features)
coef = (-1) ** idx * np.exp(-idx / 10)
coef[10:] = 0  
y = np.dot(X, coef)

y += 0.01 * np.random.normal(size=n_samples)

n_samples = X.shape[0]
ddata1 = [data1,"score"]
ddata2 = [data2,"comments number"]
ddata3 = [data3,"rank"]
print(len(ddata1[0]))
def lasso_elastic_net(data):
    y_train = data[0][0:200] 
    X_train = np.linspace(0, 199, 200).reshape(-1, 1) 
    X_test = np.linspace(201, 250, 50).reshape(-1, 1)
    y_test = data[0][200:250]
    
    alpha = 0.1
    lasso = Lasso(alpha=alpha)

    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
    r2_score_lasso = r2_score(y_test, y_pred_lasso)
    print(lasso)
    print("r^2 on test data : %f" % r2_score_lasso)


    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

    y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
    r2_score_enet = r2_score(y_test, y_pred_enet)
    print(enet)
    print("r^2 on test data : %f" % r2_score_enet)

    m, s, _ = plt.stem(np.where(enet.coef_)[0], enet.coef_[enet.coef_ != 0], markerfmt='x', label='Elastic net coefficients', use_line_collection=True)
    plt.setp([m, s], color="#2ca02c")
    m, s, _ = plt.stem(np.where(lasso.coef_)[0], lasso.coef_[lasso.coef_ != 0],markerfmt='x', label='Lasso coefficients',use_line_collection=True)
    plt.setp([m, s], color='#ff7f0e')
    plt.stem(np.where(coef)[0], coef[coef != 0], label='true coefficients',markerfmt='bx', use_line_collection=True)

    plt.legend(loc='best')
    plt.title(data[1]+" Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f"% (abs(r2_score_lasso), abs(r2_score_enet)))
    plt.show()


# In[22]:


lasso_elastic_net(ddata1)


# In[23]:


lasso_elastic_net(ddata2)


# In[24]:


lasso_elastic_net(ddata3)


# In[25]:


import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
rng = np.random.RandomState(0)

data0=np.linspace(0,249,250)

def kernal_rige_regression_SVR(datax,datay,factor,target):
    # Generate sample data
    X=datax[:100].reshape(-1,1)
    y=datay[:100]

    X_plot = np.linspace(0, 5, 100000)[:, None]
    
    # Fit regression model
    train_size = 100
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                   "gamma": np.logspace(-2, 2, 5)})

    kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                      param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                  "gamma": np.logspace(-2, 2, 5)})

    t0 = time.time()
    svr.fit(X[:train_size], y[:train_size])
    svr_fit = time.time() - t0
    print("SVR complexity and bandwidth selected and model fitted in %.3f s"
          % svr_fit)

    t0 = time.time()
    kr.fit(X[:train_size], y[:train_size])
    kr_fit = time.time() - t0
    print("KRR complexity and bandwidth selected and model fitted in %.3f s"
          % kr_fit)

    sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
    print("Support vector ratio: %.3f" % sv_ratio)

    t0 = time.time()
    y_svr = svr.predict(X_plot)
    svr_predict = time.time() - t0
    print("SVR prediction for %d inputs in %.3f s"
          % (X_plot.shape[0], svr_predict))

    t0 = time.time()
    y_kr = kr.predict(X_plot)
    kr_predict = time.time() - t0
    print("KRR prediction for %d inputs in %.3f s"
          % (X_plot.shape[0], kr_predict))

    # Look at the results
    sv_ind = svr.best_estimator_.support_
    plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
                zorder=2, edgecolors=(0, 0, 0))
    plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1,
                edgecolors=(0, 0, 0))
    plt.plot(X_plot, y_svr, c='r',
             label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
    plt.plot(X_plot, y_kr, c='g',
             label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
    plt.xlabel(factor)
    plt.ylabel(target)
    plt.title('SVR versus Kernel Ridge')
    plt.legend()

    # Visualize training and prediction time
    plt.figure()

    # Generate sample data
    X=datax[100:200].reshape(-1,1)
    y=datay[100:200]
    sizes = np.logspace(1, 4, 7).astype(int)
    for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
                                               gamma=10),
                            "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
        train_time = []
        test_time = []
        for train_test_size in sizes:
            t0 = time.time()
            estimator.fit(X[:train_test_size], y[:train_test_size])
            train_time.append(time.time() - t0)

            t0 = time.time()
            estimator.predict(X_plot[:1000])
            test_time.append(time.time() - t0)

        plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
                 label="%s (train)" % name)
        plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
                 label="%s (test)" % name)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Train size")
    plt.ylabel("Time (seconds)")
    plt.title('Execution Time')
    plt.legend(loc="best")

    # Visualize learning curves
    plt.figure()

    svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
    kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
    train_sizes, train_scores_svr, test_scores_svr =         learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)
    train_sizes_abs, train_scores_kr, test_scores_kr =         learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                       scoring="neg_mean_squared_error", cv=10)

    plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
             label="SVR")
    plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g",
             label="KRR")
    plt.xlabel("Train size")
    plt.ylabel("Mean Squared Error")
    plt.title('Learning curves')
    plt.legend(loc="best")

    plt.show()


# In[26]:


kernal_rige_regression_SVR(data0,data1,'rank','score')


# In[27]:


kernal_rige_regression_SVR(data0,data2,'rank','comments number')


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
def f(x):
    return x * np.sin(x)

def polyRegression(data1,data2,name1,name2,num):
    x_plot=np.linspace(0,num,num+1)

    x=data1
    y=data2
    X = x[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    colors = ['teal', 'yellowgreen', 'gold']
    lw = 2
    plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
             label="ground truth")
    plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")
    plt.xlabel(name1)
    plt.ylabel(name2)

    for count, degree in enumerate([2,3,4]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,label="degree %d" % degree)

    plt.legend(loc='lower left')

    plt.show()


# In[29]:


polyRegression(data0,data2,"rank","comments number",300)


# In[30]:


df1=pd.read_csv("comment1.csv")
ls1=[]
ls2=[]
for i in range(0,220):
    if(df1.loc[i,"comment"] is np.nan):
        ls1.append([0,0,0,0,0,0,0])
    else:
        ls1.append(list(emotion.emotion_count(df1.loc[i,"comment"]).values())[2:])
    ls2.append(df1.loc[i,"score"])


# In[31]:


def J(theta, X_b , y):
    try:
        return np.sum((y - X_b.dot(theta))**2) / len(X_b)
    except:
        return float('inf')
    
def dJ(theta, X_b, y):
    res = np.empty(len(theta))   
    res[0] = np.sum(X_b.dot(theta) - y)  
    for i in range(1, len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
    return res * 2 / len(X_b)


def gradient_descent(X_b, y, initial_theta, eta, n_iters = 1e3, epsilon=1e-8):
    theta = initial_theta
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        if(abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
            
        cur_iter += 1
        print("epoch"+str(cur_iter)+": theta="+str(theta))

    return theta

def model_score(ls,theta):
    score=0
    for i in range(0,7):
        score+=lsf[i]*theta[i]
    return int(score)


# In[32]:


ls1n=np.array(ls1).reshape(-1,7)
X_b = np.hstack([ls1n])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01
cnt=0
theta = gradient_descent(X_b, ls2, initial_theta, eta)
print(theta)


# In[33]:


#test1
ss="怒赞，很难有一部电影能比《肖申克的救赎》更好的诠释梦想与救赎这两个词的关联，电影予人带来心理的洗涤震撼是如此深刻，对比安迪，我们生活中看似无以能迈不过的坎又算什么？当你若能一直心拥梦想，哪怕失败，也定能获得希望的救赎。 "
lsf=list(emotion.emotion_count(ss).values())[2:]
print(model_score(lsf,theta))


# In[34]:


#test2
ss="大众经典我从不感冒，为什么？我欣赏水平不行？"
#print(list(emotion.emotion_count(ss).values())[2:])
lsf=list(emotion.emotion_count(ss).values())[2:]
print(model_score(lsf,theta))


# In[35]:


#test3
ss="轻松诙谐，很有趣的搭档故事。但叙事浮于表面，人物塑造脸谱化，故事缺乏真实感（尽管是据真事改编）。弗朗索瓦·克鲁塞和奥玛·赛的表演动人，但受剧本限制人物缺乏层次。过分中正的温情。"
lsf=list(emotion.emotion_count(ss).values())[2:]
print(model_score(lsf,theta))

