import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
                                                  #use train data
mydata = pd.read_csv("train-data.csv")
data1 = mydata.copy()

data1.drop("Unnamed: 0", axis=1, inplace=True)
data1['No._of_Years'] = 2021 - data1['Year']

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

data1.loc[:, 'Fuel_Type'] = le1.fit_transform(data1['Fuel_Type'])
data1.loc[:, 'Transmission'] = le2.fit_transform(data1['Transmission'])
data1.loc[:, 'Owner_Type'] = le3.fit_transform(data1['Owner_Type'])
data1.loc[:, 'Seats'] = le4.fit_transform(data1['Seats'])

data1['New_Price'].fillna(value='0.00 Lakh', inplace=True)
data1['Seats'].fillna(round(data1['Seats'].mean()), inplace=True)   # mean, mode, median
data1['Engine'].fillna('0 CC', inplace=True)
data1['Power'].fillna('0.01 bhp', inplace=True)
data1['Mileage'].fillna('0.01 kmpl', inplace=True)

for i in range(len(data1)):
    data1['Engine'][i] = int(data1['Engine'][i].split()[0])
    data1['Mileage'][i] = float(data1['Mileage'][i].split()[0])
    data1['New_Price'][i] = float(data1['New_Price'][i].split()[0])

    # power is having some null values -> example: 'null bhp', there are 107 values like this.
    if data1['Power'][i].split()[0] != 'null':
        data1['Power'][i] = float(data1['Power'][i].split()[0])
    elif data1['Power'][i].split()[0] == 'null':
        data1['Power'][i] = 0.0
for i in range(len(data1)):
    if data1['Engine'][i] == 0:
        data1['Engine'][i] = data1['Engine'].mean()
    if data1['Mileage'][i] == 0.01:
        data1['Mileage'][i] = data1['Mileage'].mean()
    if data1['Power'][i] == 0.01:
        data1['Power'][i] = data1['Power'].mean()
# Converting to float datatype
data1['Engine'] = data1['Engine'].astype("float64")
data1['Mileage'] = data1['Mileage'].astype("float64")
data1['Power'] = data1['Power'].astype("float64")
data1['New_Price'] = data1['New_Price'].astype("float64")

                                              #using test data
mydata2 = pd.read_csv("test-data.csv")
data2 = mydata.copy()

data2.drop("Unnamed: 0", axis=1, inplace=True)

data2['No._of_Years'] = 2021 - data2['Year']
le11 = LabelEncoder()
le22 = LabelEncoder()
le33 = LabelEncoder()
le44 = LabelEncoder()

data2.loc[:, 'Fuel_Type'] = le11.fit_transform(data2['Fuel_Type'])
data2.loc[:, 'Transmission'] = le22.fit_transform(data2['Transmission'])
data2.loc[:, 'Owner_Type'] = le33.fit_transform(data2['Owner_Type'])
data2.loc[:, 'Seats'] = le44.fit_transform(data2['Seats'])

data2['New_Price'].fillna(value='0.00 Lakh', inplace=True)
data2['Engine'].fillna('0 CC', inplace=True)
data2['Power'].fillna('0.01 bhp', inplace=True)
data2['Mileage'].fillna('0.01 kmpl', inplace=True)

for i in range(len(data2)):
    data2['Mileage'][i] = float(data2['Mileage'][i].split()[0])
    data2['New_Price'][i] = float(data2['New_Price'][i].split()[0])
    data2['Engine'][i] = float(data2['Engine'][i].split()[0])
    if data2['Power'][i].split()[0] !='null':
        data2['Power'][i] = float(data2['Power'][i].split()[0])
    elif data2['Power'][i].split()[0] =='null':
        data2['Power'][i] = 0.0

for i in range(len(data2)):
    if data2['Mileage'][i] == 0.01:
        data2['Mileage'][i] = data2['Mileage'].mean()
    if data2['Engine'][i] == 0.00:
        data2['Engine'][i] = data2['Engine'].mean()
    if data2['Power'][i] == 0.01:
        data2['Power'][i] = data2['Power'].mean()

data2['Mileage'] = data2['Mileage'].astype("float64")
data2['New_Price'] = data2['New_Price'].astype("float64")
data2['Engine'] = data2['Engine'].astype("float64")
data2['Power'] = data2['Power'].astype("float64")


#use required features

x_train = data1.drop(['Name', 'Location', 'Price', 'Year'], axis=1)
y_train = data1[['Price']]
x_test = data2.drop(['Name', 'Location', 'Year'], axis=1)
y_train = y_train.astype('int')
decision = DecisionTreeRegressor(criterion='mse', max_depth=25, min_samples_leaf=1)
decision.fit(x_train, y_train)
pickle.dump(decision, open('decisontree.pkl', 'wb'))