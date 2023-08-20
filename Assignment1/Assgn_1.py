import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input1=np.load('data/inputs_Dataset-1.npy')
input1

input1.shape

data1=pd.DataFrame(input1)

data1

output1=np.load('data/outputs_Dataset-1.npy')

data1['target']=output1



from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def prediction(one_df,n):
  #defining the value of k in kfold
  kf = KFold(n_splits=n)
  weights = np.ones(11)
  #initialising metrics array
  metrics_kfold = []

  #looping for each fold
  for train_index, test_index in kf.split(one_df):
    #Train-test split
    X_train, X_test = one_df.iloc[train_index,:-1], one_df.iloc[test_index,:-1]
    y_train, y_test = one_df.iloc[train_index,-1], one_df.iloc[test_index,-1]
    for i in range(len(X_train)):
      temp_x=X_train.iloc[i].to_numpy()
      #calculating wTx
      prod = np.dot(weights,temp_x)

      if prod>=0:
        #if our model classifies as 1, but target variable is 0
        if y_train.iloc[i]==0:
          weights = weights - temp_x
      else:
        if prod<0:
          #if our model classifies as 0, but target variable is 1
          if y_train.iloc[i]==1:
            weights = weights + temp_x

    #building prediction array for the test data
    pred = []
    for i in range(len(X_test)):
      z=np.dot(weights,X_test.iloc[i].to_numpy())
      if z>=0:
        pred.append(1)
      else:
        pred.append(0)

    #comparing prediction and test data
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    #appending metrics for each fold
    metrics_kfold.append([acc, prec, rec, f1])
  return metrics_kfold

five=prediction(data1,5)
six=prediction(data1,6)
seven=prediction(data1,7)
eight=prediction(data1,8)
nine=prediction(data1,9)

for i in range(5,10):
  val=prediction(data1,i)
  col=['accuracy','precision','recall','f1score']
  val=pd.DataFrame(val,columns=col)
  print(f"The metrics for K={i} are-:")
  print('\n')
  print(val)
  print('\n')
  mean = val.mean()
  print(f"The mean of metrics for K={i} are- :")
  print('\n')
  print(mean)
  print('\n')
  vari = val.var()
  print(f"The varaince of metrics for K={i} are- :")
  print('\n')
  print(vari)
  print('\n')

x=0.8*len(data1)
x=int(x)

X_train, X_test = data1.iloc[:x,:-1], data1.iloc[x:,:-1]
y_train, y_test = data1.iloc[:x,-1], data1.iloc[x:,-1]

misclassed = []
weights = np.ones(11)
epochs=2000
for k in range(epochs):

  for i in range(len(X_train)):
    temp_x=X_train.iloc[i].to_numpy()
    prod = np.dot(weights,temp_x)
    if prod>=0:
      if y_train.iloc[i]==0:
        weights = weights - temp_x
    else:
      if prod<0:
        if y_train.iloc[i]==1:
          weights = weights + temp_x
  cnt=0
  for j in range(len(X_test)):
    temp_x=X_test.iloc[j].to_numpy()
    prod = np.dot(weights,temp_x)
    if prod>=0:
       if y_test.iloc[j]==0:
         cnt=cnt+1
    else:
       if prod<0:
         if y_test.iloc[j]==1:
           cnt=cnt+1
  misclassed.append(cnt)



iterations = range(1,epochs+1)
plt.figure(figsize=(25, 6))
plt.plot(iterations, misclassed)
plt.xlabel('Iterations')
plt.ylabel('Misclassified Instances')
plt.title('Misclassified Instances vs. Iterations')
plt.show()

input2=np.load('data/inputs_Dataset-2.npy')
data2=pd.DataFrame(input2)
output2=np.load('data/outputs_Dataset-2.npy')
data2['target']=output2

x=0.8*len(data2)
x=int(x)
X_train, X_test = data2.iloc[:x,:-1], data2.iloc[x:,:-1]
y_train, y_test = data2.iloc[:x,-1], data2.iloc[x:,-1]

misclassed = []
metrics = []
weights = np.ones(11)
epochs=2000
mi=1000
best_weight=weights
for k in range(epochs):

  for i in range(len(X_train)):
    temp_x=X_train.iloc[i].to_numpy()
    prod = np.dot(weights,temp_x)
    if prod>=0:
      if y_train.iloc[i]==0:
        weights = weights - temp_x
    else:
      if prod<0:
        if y_train.iloc[i]==1:
          weights = weights + temp_x
  cnt=0
  for j in range(len(X_test)):
    temp_x=X_test.iloc[j].to_numpy()
    prod = np.dot(weights,temp_x)
    if prod>=0:
       if y_test.iloc[j]==0:
         cnt=cnt+1
    else:
       if prod<0:
         if y_test.iloc[j]==1:
           cnt=cnt+1
  mi=min(mi,cnt)
  if cnt==mi:
    best_weight=weights
  misclassed.append(cnt)

# pred=np.dot(best_weight,X_test)
pred=[]
metrics=[]
for i in range(len(X_test)):
  val=np.dot(best_weight,X_test.iloc[i].to_numpy())
  if val>=0:
    pred.append(1)
  else:
    pred.append(0)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
metrics.append([acc, prec, rec, f1])

iterations = range(1,epochs+1)
plt.figure(figsize=(25, 6))
plt.plot(iterations, misclassed)
plt.xlabel('Iterations')
plt.ylabel('Misclassified Instances')
plt.title('Misclassified Instances vs. Iterations')
plt.show()

metrics=pd.DataFrame(metrics,columns=col)

print("The performance metrics on the test data of dataset2 are")
print('\n')
print(metrics)
print('\n')

input3=np.load('data/inputs_Dataset-3.npy')
data3=pd.DataFrame(input3)
output3=np.load('data/outputs_Dataset-3.npy')
data3['target']=output3

x=0.8*len(data3)
x=int(x)
X_train, X_test = data3.iloc[:x,:-1], data3.iloc[x:,:-1]
y_train, y_test = data3.iloc[:x,-1], data3.iloc[x:,-1]

misclassed = []
metrics = []
weights = np.ones(11)
epochs=2000
mi=1000
best_weight=weights
for k in range(epochs):

  for i in range(len(X_train)):
    temp_x=X_train.iloc[i].to_numpy()
    prod = np.dot(weights,temp_x)
    if prod>=0:
      if y_train.iloc[i]==0:
        weights = weights - temp_x
    else:
      if prod<0:
        if y_train.iloc[i]==1:
          weights = weights + temp_x
  cnt=0
  for j in range(len(X_test)):
    temp_x=X_test.iloc[j].to_numpy()
    prod = np.dot(weights,temp_x)
    if prod>=0:
       if y_test.iloc[j]==0:
         cnt=cnt+1
    else:
       if prod<0:
         if y_test.iloc[j]==1:
           cnt=cnt+1
  mi=min(mi,cnt)
  if cnt==mi:
    best_weight=weights
  misclassed.append(cnt)

iterations = range(1,epochs+1)
plt.figure(figsize=(25, 6))
plt.plot(iterations, misclassed)
plt.xlabel('Iterations')
plt.ylabel('Misclassified Instances')
plt.title('Misclassified Instances vs. Iterations')
plt.show()

# pred=np.dot(best_weight,X_test)
pred=[]
metrics=[]
for i in range(len(X_test)):
  val=np.dot(best_weight,X_test.iloc[i].to_numpy())
  if val>=0:
    pred.append(1)
  else:
    pred.append(0)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
metrics.append([acc, prec, rec, f1])

metrics=pd.DataFrame(metrics,columns=col)
print("The performance metrics on the test data of dataset3 are")
print('\n')
print(metrics)
print('\n')

