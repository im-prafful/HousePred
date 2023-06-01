# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('train.csv')
x = pd.DataFrame(dataset.iloc[:, [1, 4, 5, 6]].values)
y = pd.DataFrame(dataset.iloc[:, 2].values)
city = pd.DataFrame(dataset.iloc[:, 4].values)

# Handling Missing Data
x.ffill(axis=1)
# Encoding Categorical Data
city = pd.get_dummies(city)

x = pd.DataFrame(dataset.iloc[:, [1, 5, 6]].values)
city = city.fillna(0)
frames = [x, city]
result = pd.concat(frames, axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
temp = result.iloc[:, 1]
temp = np.asarray(temp)
temp = temp.reshape(-1, 1)
result.iloc[:, 1] = sc.fit_transform(temp)
#results.rename(columns = {0:"rooms",1:"sqft",2:"toilets"}, inplace = True)

# Performing MLR on the dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(result, y)

# Comparing Results
y_pred = lr.predict(result)
np.set_printoptions(precision=0)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y.reshape(len(y), 1)), 1))

# Plotting Results
plt.scatter(result, y, color='red')
plt.plot(result, y_pred, color='blue')
plt.title('Mine the Model')
plt.xlabel('Features')
plt.ylabel('Expected Price')
