import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

dir = 'D:\selected project\Data_Set'

categories = ['Jaguar','Leopard']

data =[]



for category in categories:
    path = os.path.join(dir, category)
    label=categories.index(category)
    
    
    
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        pet_img=cv2.imread(imgpath,0)
        
        
        try:
            pet_img=cv2.resize(pet_img,(50,50))
            image = np.array(pet_img).flatten()
        
        
            data.append([image,label])
            
        except Exception as e:
            pass
    
        
    

#print(len(data))
  
 
pick_in =open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()


pick_in =open('data1.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()




random.shuffle(data)
features = []
labels = []



for feature , label in data:
    features.append(feature)
    labels.append(label)
    
    
    
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25)

model = SVC(C=1,kernel='poly',gamma='auto')
#model = SVC(C=1, kernel='rbf', gamma=0.00001)
model.fit(xtrain, ytrain)




pick = open('model.sav' , 'wb')
pickle.dump(model,pick)
pick = open('model.sav' , 'rb')
model = pickle.load(pick)
pick.close()



prediction=model.predict(xtest)
accuracy = model.score(xtest, ytest)



categories = ['Jaguar','Leopard']
#print('Accuracy :', accuracy)
print('prediction is :', categories[prediction[0]])




mypet=xtest[0].reshape(50,50)
plt.imshow(mypet,cmap='gray')
plt.show()   





###################################################################



# preds = model.predict(xtest)
# targs = ytest 
# print("accuracy: ", metrics.accuracy_score(targs, preds))
# print("precision: ", metrics.precision_score(targs, preds)) 
# print("recall: ", metrics.recall_score(targs, preds))
# print("f1: ", metrics.f1_score(targs, preds))
# print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))
# train_preds = preds



###################################################################


preds = model.predict(xtrain)
targs = ytrain 
print("accuracy: ", metrics.accuracy_score(targs, preds))
print("precision: ", metrics.precision_score(targs, preds)) 
print("recall: ", metrics.recall_score(targs, preds))
print("f1: ", metrics.f1_score(targs, preds))
print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))
test_preds = preds



#####################################################################


ytrain_pred = model.decision_function(xtrain)    
ytest_pred = model.decision_function(xtest) 

train_fpr, train_tpr, tr_thresholds = roc_curve(ytrain, ytrain_pred)
test_fpr, test_tpr, te_thresholds = roc_curve(ytest, ytest_pred)

plt.grid()

plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()



