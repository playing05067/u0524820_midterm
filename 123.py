
# coding: utf-8

# In[25]:




# In[26]:

'''
from urllib2 import urlopen
from contextlib import closing
url = "http://aima.cs.berkeley.edu/data/iris.csv"
with closing(urlopen(url)) as u, open("iris.csv", "w") as f:
 f.write(u.read())
'''


# In[27]:

import os 
os.getcwd()


# In[28]:

os.listdir('/Users/apple')


# In[29]:

from numpy import genfromtxt, zeros
# read the first 4 columns
data = genfromtxt("iris.csv",delimiter=",",usecols=(0,1,2,3))
# read the fifth column
# this fifth column is the label (or target)
target = genfromtxt("iris.csv",delimiter=",",usecols=(4),dtype=str)


# In[30]:

print data.shape
print target.shape


# In[31]:

print set(target)


# In[32]:

t = zeros(len(target))
t[target == "setosa"] = 1
t[target == "versicolor"] = 2
t[target == "virginica"] = 3


# In[33]:

# import 你的演算法
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[34]:

# 建構你的訓練模型並且餵入資料集
LinearDiscrimnat_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(data, t)


# In[35]:

print LinearDiscrimnat_clf.predict(data[0])


# In[36]:

print t[0]


# In[37]:

from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data, t,test_size=0.4, random_state=0)


# In[38]:

LinearDiscrimnat_clf.fit(train,t_train) # train
print LinearDiscrimnat_clf.score(test,t_test) # test


# In[39]:

from sklearn.metrics import confusion_matrix
print confusion_matrix(LinearDiscrimnat_clf.predict(test),t_test)


# In[40]:

from sklearn.metrics import classification_report
print classification_report(LinearDiscrimnat_clf.predict(test), t_test,target_names=["setosa", "versicolor", "virginica"])


# In[41]:

from sklearn.cross_validation import cross_val_score
# cross validation with 6 iterations
scores = cross_val_score(LinearDiscrimnat_clf, data, t, cv=6)
print scores


# In[42]:

from numpy import mean
print mean(scores)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



