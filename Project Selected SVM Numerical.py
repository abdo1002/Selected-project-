import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2 , f_classif
from sklearn.preprocessing import StandardScaler



Depression = pd.read_csv('b_depressed.csv')
Depression.head()
Depression.shape
Depression.isnull().sum()
Depression = Depression.dropna()
Depression.isnull().sum()

X = Depression.drop(columns=['Survey_id','Ville_id' , 'depressed'],axis=1)
Y = Depression['depressed']

FeatureSelection = SelectPercentile (score_func = chi2 , percentile = 30 )
X = FeatureSelection.fit_transform(X,Y)


scaler = StandardScaler(copy = True , with_mean = True , with_std=True)
X = scaler.fit_transform(X)
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,stratify=Y,random_state=2)


classifier = svm.SVC(kernel='linear')

classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print(training_data_accuracy)



## roc curve From matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, plot_roc_curve

plot_roc_curve(classifier,X_test,Y_test)
plt.show()
Y_test_pred = classifier.predict(X_test)
print("Accuracy: ", accuracy_score(Y_test, Y_test_pred))




plot_roc_curve(classifier,X_train,Y_train)
plt.show()
Y_train_pred = classifier.predict(X_train)
print("Accuracy: ", accuracy_score(Y_train, Y_train_pred))


##Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy_score(Y_test, Y_pred)

