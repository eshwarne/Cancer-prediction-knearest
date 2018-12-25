import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd;
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.drop('id',1,inplace=True)
print(df.columns)
#make the NaN instances as outliers
df.replace('?',-99999,inplace=True)

# make feature vector and label vector
X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.3)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train,Y_train)
accuracy=classifier.score(X_test,Y_test)
print(accuracy)

#make a prediction on unknown data
X_predict = np.array([8.5,7,7,6,4,10,4,1,2])
X_predict = X_predict.reshape(1,-1) 
prediction=classifier.predict(X_predict)
if(4 in prediction):
    print("Malignant")
else:
    print("Benign")


