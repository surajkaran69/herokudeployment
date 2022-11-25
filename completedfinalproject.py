#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel(r"D:\project82dataset.xlsx")
import numpy as np


# In[2]:


df.isnull().sum()


# In[3]:


col=['Age','StdWorkingHrsPerDay','Pulse_rate_in_idle','Average_Pulse_rate_in_activity','Calories_Burnt_per_shift','TargetedWorkPerWorkingDay','Actual_Workdone','PendingWork']
for c in col:
    plt.boxplot(df[col])


# In[4]:


df.info()


# In[5]:


df


# In[6]:


#lavel encoding
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()


# In[7]:


df['Activity_Category']=le.fit_transform(df['Activity_Category'])
df['Productivity']=le.fit_transform(df['Productivity'])


# In[8]:


df = df.drop('Id_number', axis=1)
df = df.drop('Name_of_worker', axis=1)


# In[9]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV


# In[10]:


X=df.drop('Productivity',axis=1)


# In[11]:


y=df['Productivity']


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


lr=LogisticRegression()


# In[15]:


lr.fit(X_train,y_train)


# In[16]:


lr.coef_


# In[17]:


lr.intercept_


# In[18]:


lr.score(X_test,y_test)
y_pred = lr.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[20]:


knn.fit(X_train,y_train)


# In[21]:


knn.score(X_test,y_test)


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix


# In[23]:


y_pred = knn.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(f'Train_accuracy= {knn.score(X_train,y_train):.3f}')
print(f'Test_accuracy= {knn.score(X_test,y_test):.3f}')


# In[24]:


para = {'max_depth':(10,30,50,70,90,100)
       , "criterion":('gini','entropy')
       , 'max_depth' : (3,5,7,9,10)
       , 'max_features':('auto','sqrt','log2')
       , 'min_samples_split':(2,4,6)
       }
from sklearn.tree import DecisionTreeClassifier
DT_grid = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions= para,cv=5, verbose=True)


# In[25]:


DT_grid.fit(X_train,y_train)


# In[26]:


DT_grid.best_estimator_


# In[29]:


model = DecisionTreeClassifier(    criterion='entropy', max_depth=3, max_features='log2', min_samples_split=4    )
model.fit(X_train,y_train)


# In[30]:


model.score(X_test,y_test)


# In[31]:


print(f'Train_accuracy= {model.score(X_train,y_train):.3f}')
print(f'Test_accuracy= {model.score(X_test,y_test):.3f}')


# In[32]:


y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[33]:


df.columns


# In[34]:


import pickle


# In[35]:


from pandas_profiling import ProfileReport


# In[36]:


profile = ProfileReport(df, title="Pandas Profiling Report")


# In[37]:


profile


# In[38]:


from sklearn.svm import SVC


# In[39]:


SVM = SVC(kernel='rbf', C=100, random_state=10).fit(X_train,y_train)


# In[40]:


SVM.fit(X_train, y_train)


# In[41]:


y_pred = SVM.predict(X_test)


# In[42]:


svm_train_acc = classification_report(y_train, SVM.predict(X_train))
svm_test_acc = classification_report(y_test, y_pred)


# In[43]:


print(f"Training Accuracy of svm : {svm_train_acc}")
print(f"Test Accuracy of svm : {svm_test_acc}")
print(SVM.score(X_test,y_test))


# In[44]:


from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier(criterion = 'gini', max_depth = 8, max_features = 'sqrt', min_samples_leaf = 4, min_samples_split = 5, n_estimators = 150)
rc.fit(X_train, y_train)


# In[45]:


y_pred = rc.predict(X_test)


# In[46]:


train_acc = classification_report(y_train, rc.predict(X_train))
test_acc = classification_report(y_test, y_pred)


# In[47]:


print(f"Training Accuracy of Random Forest Classifier:{train_acc}")
print(f"Test Accuracy of Random Forest Classifier: {test_acc}")
print(rc.score(X_test,y_test))


# In[48]:


from sklearn.neural_network import MLPClassifier


# In[49]:


ANN = MLPClassifier(hidden_layer_sizes=(100,100,100),batch_size=10,learning_rate_init=0.01,max_iter=2000,random_state=10)


# In[50]:


ANN.fit(X_train,y_train)


# In[51]:


y_pred = ANN.predict(X_test)


# In[52]:


ann_train_acc = classification_report(y_train, ANN.predict(X_train))
ann_test_acc = classification_report(y_test, y_pred)


# In[53]:


print(f"Training Accuracy of ANN Model: {ann_train_acc}")
print(f"Test Accuracy of ANN Model: {ann_train_acc}")
print(ANN.score(X_test,y_test))


# In[54]:


ANN.score(X_test,y_test)


# In[56]:


p = open('model.pkl','wb')
pickle.dump(model,p)
p.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




