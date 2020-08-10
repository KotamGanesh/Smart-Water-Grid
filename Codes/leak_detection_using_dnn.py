from google.colab import drive
drive.mount('/content/gdrive')

import keras
import numpy as np
import pandas as pd

from google.colab import files
file = files.upload()

x_train = pd.read_excel('x_train.xlsx')
y_train = pd.read_excel('y_train.xlsx')
x_test = pd.read_excel('x_test.xlsx')
y_test = pd.read_excel('y_test.xlsx')


model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(9,)))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=40)

print(model.evaluate(x=x_train, y=y_train))
model.metrics_names

print(model.evaluate(x=x_test, y=y_test))
model.metrics_names

Y_pred = model.predict(x_test)
Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred ]
print(Y_pred)

combinedDF = pd.concat([x_train, y_train], axis=1)
combinedDF

total = 0
correct = 0
wrong = 0
wrong_prediction_df = pd.DataFrame(columns = ['Junc 10', 'Junc 11', 'Junc 12', 'Junc 13', 'Junc 21', 'Junc 22', 'Junc 23', 'Junc 31', 'Junc 32', 'condition']) 
for i in range(len(Y_pred)):
  total=total+1
  if(y_test['condition'][i] == Y_pred[i]):
    correct=correct+1
  else:
    wrong=wrong+1
    currentPressureArray = x_test.loc[i].to_list()
    currentPressureArray.append(y_test['condition'][i])
    wrong_prediction_df = wrong_prediction_df.append(pd.Series(currentPressureArray, index = combinedDF.columns), ignore_index=True)


print("Total " + str(total))
print("Correct " + str(correct))
print("Wrong " + str(wrong))
print(wrong_prediction_df)

"""DATA VISUALISATION

THE TRAIN SET PRESSURE DATA HAS BEEN PLOTTED FOR LEAK AND NON-LEAK CASE
"""

import matplotlib.pyplot as plt
for col in combinedDF:  
    if col != 'condition':
        plt.scatter('condition', col , data=combinedDF)
        plt.ylabel(col +  'pressure')
        plt.xlabel('0=no leak, 1=leak')
        plt.axis([-0.1, 1.1, 0, 140])
        plt.show()

"""DATA VISUALISATION:

THE FOLLOWING GRAPHS SHOW THE PRESSURES WHICH WERE WRONGLY PREDICTED
"""

for col in wrong_prediction_df:  
    if col != 'condition':
        plt.scatter('condition', col , data=wrong_prediction_df)
        plt.ylabel(col +  'pressure')
        plt.xlabel('0=no leak, 1=leak')
        plt.axis([-0.1, 1.1, 0, 140])
        plt.show()

