import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dir_path = '../pythonProject//'

# define column names for easy indexing
index_names = ['dataset_id','unit_id', 'cycle']
setting_names = ['setting 1', 'setting 2', 'setting 3']
sensor_names = ['sensor {}'.format(i) for i in range(1,22)]
# col_names = index_names + setting_names + sensor_names

# read data
train = pd.read_csv((dir_path+'train.csv'))
test = pd.read_csv((dir_path+'test.csv'))
y_test = pd.read_csv((dir_path+'RUL.csv'))

print(train.columns)
print(y_test )


def URL(df):
    grouped_by_unit=df.groupby(by='unit_id')
    max_cycle=grouped_by_unit["cycle"].max()
    result_frame=df.merge(max_cycle.to_frame(name='max_cycle'),left_on='unit_id',right_index=True)
    print(result_frame)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["cycle"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


train = URL(train)
print(train[['unit_id'] +['cycle']+ ['RUL']].head())




def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))

print(train)

drop_sensors = ['sensor 1','sensor 5','sensor 6','sensor 10','sensor 16','sensor 18','sensor 19']
drop_labels = index_names + setting_names + drop_sensors

X_train = train.drop(drop_labels, axis=1)
y_train = X_train.pop('RUL')
X_test = test.groupby('unit_id').last().reset_index().drop(drop_labels, axis=1)
print(X_train,y_train)

lm = LinearRegression()
lm.fit(X_train, y_train)
# predict and evaluate
y_hat_train = lm.predict(X_train)
evaluate(y_train, y_hat_train, 'train')

y_hat_test = lm.predict(X_test)
evaluate(y_test, y_hat_test)