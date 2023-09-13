

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

df=pd.read_csv('data\dataset.csv')


def prepare(data):
  col=data.columns
  #dropping features with more than 50% null values
  cols=[c for c in col if data[c].isnull().sum()/len(data)>0.5]
  data.drop(columns=cols,inplace=True)
  #filling null values, numeric with mean and object with forward fill
  for c in data.columns:
    if data[c].dtype == 'object':
      data[c].fillna(method='ffill',inplace=True)
    else:
      data[c].fillna(data[c].mean(),inplace=True)
  return(data)

df=prepare(df)
#dropping non-feature columns
df=df.drop('User_ID',axis=1)
df=df.drop('Product_ID',axis=1)

df

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (5, 3)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

#Plotting distribution of age groups
df_temp=df
df_temp['start_age']=df_temp['Age'].apply(lambda x: int(x[:2]) if x[2]=='+' else int(x.split('-')[0]))
df_temp=df_temp.sort_values(by='start_age')
fig = px.histogram(df_temp,
                   x='Age',
                   marginal='box',
                   nbins=7,
                   title='Distribution of Age')
fig.update_layout(bargap=0.1)
fig.show()

#Plotting distribution of occupation
df_temp=df
df_temp=df_temp.sort_values(by='Occupation')
fig = px.histogram(df_temp,
                   x='Occupation',
                   marginal='box',
                   nbins=50,
                   title='Distribution of Occupation')
fig.update_layout(bargap=0.01)
fig.show()

#Plotting Distribution of Gender
fig = px.histogram(df,
                   x='Gender',
                   title='Distribution of Gender')
fig.show()

df.drop('start_age',axis=1)




#Plotting Distribution of City Category
temp_df=df
temp_df=temp_df.sort_values(by='City_Category',key=lambda x:x.str.lower())
fig=px.histogram(temp_df,
                 x='City_Category',
                 marginal='box',
                 title='City Category Distribution'
                 )
fig.update_layout(bargap=0.1)
fig.show()

#Plotting Distribution of Stay in current city
temp_df=df
temp_df['year']=temp_df['Stay_In_Current_City_Years'].apply(lambda x:int(x[0]))
temp_df=temp_df.sort_values(by='year')
fig=px.histogram(temp_df,
                 x='Stay_In_Current_City_Years',
                 marginal='box',
                 title='City Category stay Distribution'
                 )
fig.update_layout(bargap=0.1)
fig.show()

df=df.drop('year',axis=1)
df=df.drop('start_age',axis=1)

#Plotting Distribution of Marital Status
df['Marital_Status']=df['Marital_Status'].apply(lambda x:str(x))
fig = px.histogram(df,
                   x='Marital_Status',
                   nbins=2,
                   title='Distribution of Marital Status')
fig.update_layout(bargap=0.1)
fig.show()

df['Marital_Status']=df['Marital_Status'].apply(lambda x:int(x))
df.dtypes

#Plotting Distribution of Product Category 1
fig = px.histogram(df,
                   x='Product_Category_1',
                   marginal='box',
                   nbins=20,
                   title='Distribution of Product Category 1')
fig.update_layout(bargap=0.0)
fig.update_traces(marker=dict(line=dict(width=2)), selector=dict(type='histogram'))
fig.show()

#Plotting Distribution of Product Category 2
fig = px.histogram(df,
                   x='Product_Category_2',
                   marginal='box',
                   nbins=20,
                   title='Distribution of Product Category 2')
fig.update_traces(marker=dict(line=dict(width=2)), selector=dict(type='histogram'))
fig.show()

#Plotting Distribution of Purchase
fig = px.histogram(df,
                   x='Purchase',
                   marginal='box',
                   nbins=20,
                   title='Distribution of Purchase Cost')
fig.update_layout(bargap=0.0)
fig.update_traces(marker=dict(line=dict(width=2)), selector=dict(type='histogram'))
fig.show()



#Making Categorical data numeric through mapping
sex_codes = {'F':0, 'M':1}
df['Gender'] = df['Gender'].map(sex_codes)


#Making Categorical data numeric trough one hot encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(df[['City_Category']])


region_one_hot = enc.transform(df[['City_Category']])

df[['A', 'B', 'C']] = region_one_hot.toarray()

df.drop('City_Category',axis=1,inplace=True)

enc = OneHotEncoder()
enc.fit(df[['Age']])
region_one_hot = enc.transform(df[['Age']])


df[['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']] = region_one_hot.toarray()

df.drop('Age',axis=1,inplace=True)


enc = OneHotEncoder()
enc.fit(df[['Stay_In_Current_City_Years']])
region_one_hot = enc.transform(df[['Stay_In_Current_City_Years']])
enc.categories_

region_one_hot = enc.transform(df[['Stay_In_Current_City_Years']])
df[['0', '1', '2', '3', '4+']] = region_one_hot.toarray()
df.drop('Stay_In_Current_City_Years',axis=1,inplace=True)


matplotlib.rcParams['figure.figsize'] = (25, 15)
sns.heatmap(df.corr(), cmap='Reds', annot=True)
plt.title('Correlation Matrix');

def LIN_MODEL_CLOSED(X,y):
  #product of pseudo inverse and training data
  weights = np.linalg.pinv(X_train).dot(y_train)
  return weights

from sklearn.model_selection import train_test_split

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

X = df.drop(columns=['Purchase']) #features
y = df['Purchase'] #labels
X = np.c_[np.ones(X.shape[0]), X] # Adding additional feature for bias
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, shuffle=True, random_state=42)
X.shape, X_train.shape, X_test.shape

weights=LIN_MODEL_CLOSED(X_train,y_train)


y_pred=X_test.dot(weights)

print(f"The test loss for unscaled data model is {rmse(y_test, y_pred)}")

"""## Scaling"""

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_test=scaler.transform(X_test)
X_train=scaler.transform(X_train)



weights=LIN_MODEL_CLOSED(X_train,y_train)



y_pred=X_test.dot(weights)

print(f"The test loss for scaled data model is {rmse(y_test, y_pred)}")



#Assign3

X = df.drop(columns=['Purchase']) #features
y = df['Purchase'] #labels
# X = np.c_[np.ones(X.shape[0]), X]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Scaling
scaler=StandardScaler()
scaler.fit(X_train)
X_test=scaler.transform(X_test)
X_train=scaler.transform(X_train)
X_val=scaler.transform(X_val)






#Initialiasing Weights and Bias
weights=np.ones(X_test.shape[1])
weight_orig=np.ones(X_test.shape[1])
bias=1

bias_orig=bias

learning_rate = [0.00001,0.0001,0.001,0.05,0.1]
minibatch=256
epochs=50

validation_losses=[]
test_losses=[]
weight_lr=[]
bias_lr=[]
for lr in learning_rate:
  # test_loss=[]
  weights=weight_orig
  bias=bias_orig
  validation_loss=[]
  for epoch in range(epochs):
    shuffled_indices = np.random.permutation(len(X_train))
    for batch_start in range(0, len(X_train), minibatch):
        batch_indices = shuffled_indices[batch_start:batch_start+minibatch]
        X_batch = X_train[batch_indices]
        y_batch = y[batch_indices]
        # print(weights)
        # print(X_batch)
        y_pred = X_batch.dot(weights) + bias
        error=y_pred-y_batch
        #Gradient for weights and bias
        gradient_w = (2 / len(y_batch)) * X_batch.T.dot(error)
        gradient_b = (2 / len(y_batch)) * np.sum(error)
        #Updating weights and Bias
        weights -= lr * gradient_w
        bias -= lr*gradient_b

    #validation loss (MSE) at the end of each iteration
    y_val_pred = X_val.dot(weights)+bias
    val_loss = np.mean((y_val_pred - y_val)**2)
    validation_loss.append(val_loss)
  y_test_pred = X_test.dot(weights)+bias
  test_loss = np.mean((y_test_pred - y_test)**2)
  test_losses.append(test_loss)
  validation_losses.append(validation_loss)
  weight_lr.append(weights)
  bias_lr.append(bias)

#Finding Best Learning Rate
best_lr_grad = learning_rate[np.argmin(test_losses)]
best_ind= learning_rate.index(best_lr_grad)
weights_g = weight_lr[best_ind]
bias_g = bias_lr[best_ind]

plt.figure(figsize=(10, 6))
for i, lr in enumerate(learning_rate):
    plt.plot(validation_losses[i], label=f'Learning Rate {lr}')

plt.xlabel('Iteration')
plt.ylabel('Validation Loss (MSE)')
plt.title('Validation Loss vs. Iteration for Different Learning Rates')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(learning_rate, test_losses, label='MSE vs. LR', marker='o')

plt.xlabel('Learning Rate')
plt.ylabel('Test Loss (MSE)')
plt.title('MSE Loss vs. Learning Rate')
plt.legend()
plt.grid(True)
plt.show()
print(f"Best learning rate: {best_lr_grad}")

#Assign4


X = df.drop(columns=['Purchase']) #features
y = df['Purchase'] #labels
# X = np.c_[np.ones(X.shape[0]), X]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#Scaling
scaler=StandardScaler()
scaler.fit(X_train)
X_test=scaler.transform(X_test)
X_train=scaler.transform(X_train)
X_val=scaler.transform(X_val)

from sklearn.linear_model import Ridge
alpha_values = np.arange(0.1, 1.1, 0.1)

# Initialize arrays to store MSE for each alpha value
mse_values = []
models=[]
# Train Ridge regression models with different alpha values
for alpha in alpha_values:
    # Create and fit Ridge regression model
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Make predictions on validation data
    y_val_pred = model.predict(X_val)

    # Calculate MSE on validation data
    mse = rmse(y_val, y_val_pred)
    mse_values.append(mse)
    models.append(model)

# Plot MSE vs. alpha
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, mse_values, marker='o')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Validation MSE')
plt.title('Validation MSE vs. Alpha (Ridge Regression)')
plt.grid(True)
plt.show()

# Find the best alpha value with the lowest MSE
best_alpha_ridge = alpha_values[np.argmin(mse_values)]
best_ind=np.where(alpha_values==best_alpha_ridge)
# print(best_ind[0][0])
best_ridge_model=models[best_ind[0][0]]
print(f"Best Alpha: {best_alpha_ridge}")

#Experiment 5
#For Closed Model
X = df.drop(columns=['Purchase']) #features
y = df['Purchase'] #labels
# X = np.c_[np.ones(X.shape[0]), X]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
X_val = np.c_[np.ones(X_val.shape[0]), X_val]

wts=LIN_MODEL_CLOSED(X_train,y_train)
y_pred=X_test.dot(wts)
mse=np.mean((y_pred - y_test)**2)
print(f"MSE Loss of LIN_MODEL_CLOSED is {mse}")

#For Gradient Descent Model
X = df.drop(columns=['Purchase']) #features
y = df['Purchase'] #labels
# X = np.c_[np.ones(X.shape[0]), X]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#Scaling
scaler=StandardScaler()
scaler.fit(X_train)
X_test=scaler.transform(X_test)
X_train=scaler.transform(X_train)
X_val=scaler.transform(X_val)

y_pred=X_test.dot(weights_g)+bias_g
mse=np.mean((y_pred - y_test)**2)
print(f"MSE Loss of LIN_MODEL_GRAD is {mse}")

#For Ridge Regression Model
y_pred=best_ridge_model.predict(X_test)
mse=np.mean((y_pred - y_test)**2)
print(f"MSE Loss of LIN_MODEL_RIDGE is {mse}")

