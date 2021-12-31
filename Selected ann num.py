#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# In[24]:


data = pd.read_csv(r'C:\Users\LAPTOP\Documents\b_depressed.csv')


# In[25]:


data.head(3)


# In[26]:


data.tail(3)


# In[27]:


X = data.iloc[:,3:-1].values


# In[28]:


Y = data.iloc[:,-1].values


# In[29]:


print(X)
print(Y)


# In[30]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[31]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[32]:


ann = tf.keras.models.Sequential()


# In[33]:


ann.add(tf.keras.layers.Dense(units=6,activation="relu"))


# In[34]:


ann.add(tf.keras.layers.Dense(units=6,activation="relu"))


# In[35]:


ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))


# In[36]:


myModel = tf.keras.Sequential([
    tf.keras.layers.Dense(1000,activation=tf.nn.relu,input_shape=(19900,)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# In[37]:


ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


# In[38]:


ann.fit(X_train,Y_train,batch_size=32,epochs = 100)


# In[41]:


print(ann.predict(sc.transform([[0,926,91,1,28,1,4,10,5,28912201,22861940,0,0,0,30028818,31363432,0,28411718,28292707.0]])) > 0.5)


# In[42]:


Y_pred = ann.predict(X_test)
Y_pred = (Y_pred > 0.5)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))


# In[43]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy_score(Y_test, Y_pred)


# In[44]:


history = ann.fit(X_train, Y_train, epochs=50, batch_size=5, verbose=1, validation_split=0.2)

val_loss, val_acc = ann.evaluate(X_test, Y_test)

predictions = ann.predict(X_test)

ann.summary()
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[45]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
X, Y = make_classification(n_samples=1000, n_classes=2, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=2)
ns_probs = [0 for _ in range(len(Y_test))]
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, Y_train)
lr_probs = model.predict_proba(X_test)
lr_probs = lr_probs[:, 1]
ns_auc = roc_auc_score(Y_test, ns_probs)
lr_auc = roc_auc_score(Y_test, lr_probs)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

