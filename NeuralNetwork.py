import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


finalData = pd.read_csv("finalData.csv")

finalData = finalData.drop(['Unnamed: 0'], axis = 1)

finalData['roofType_d_Fibercement'] = (finalData['roofType_d_Fibercement herunder asbest'] + finalData['roofType_d_Fibercement uden asbest'])
finalData['roofType_d_Tagpap'] = (finalData['roofType_d_Tagpap med lille hældning'] + finalData['roofType_d_Tagpap med stor hældning'])

finalData = finalData.drop(['roofType_d_Betontagsten', 'roofType_d_Fibercement herunder asbest',
                'roofType_d_Fibercement uden asbest', 'roofType_d_Glas',
                'roofType_d_Levende tage', 'roofType_d_Metal', 'roofType_d_Stråtag',
                'roofType_d_Tagpap med lille hældning',
                'roofType_d_Tagpap med stor hældning', 'latitude_b', 'longitude_b'], axis = 1)


## IMPUTING MONTHLY EXPENSES
def kvm_grp(x):
    if x <= 35:
        return '<35'
    if x <= 45:
        return '35-45'
    if x <= 55:
        return '45-55'
    if x <= 65:
        return '55-65'
    if x <= 75:
        return '65-75'
    if x <= 85:
        return '75-85'
    if x <= 95:
        return '85-95'
    if x <= 110:
        return '95-110'
    if x <= 130:
        return '110-130'
    if x <= 150:
        return '130-150'
    if x <= 175:
        return '150-175'
    if x <= 200:
        return '175-200'
    if x <= 250:
        return '200-250'
    if x <= 300:
        return  '250-300'
    else:
        return '300+'

finalData['kvm_grp'] = finalData['areaWeighted_bd'].apply(kvm_grp)

def impute_numerical(categorical_column, numerical_column):
    frames = []
    for i in list(set(finalData[categorical_column])):
        df_category = finalData[finalData[categorical_column]== i]
        if len(df_category) > 1:
            df_category[numerical_column].fillna(df_category[numerical_column].mean(),inplace = True)
        else:
            df_category[numerical_column].fillna(finalData[numerical_column].mean(),inplace = True)
        frames.append(df_category)
        final_df = pd.concat(frames)
    return final_df

finalData = impute_numerical('kvm_grp', 'mhtlExpens_b')
finalData.info()
finalData = finalData.drop(['kvm_grp'], axis = 1)

###################################################################################
###################################################################################


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

finalDataLog = finalData.copy()

finalDataLog['log_price_b'] = np.log(finalDataLog['price_b'])

X = finalDataLog.drop(['price_b', 'log_price_b', 'areaResidential_bd', 'AVM_price_d',
                       'quarter_b', 'quarter_numeric', 'indicatorSalesP_own',
                       'salesPeriod_b', 'energyMark_b', 'rebuildYear_bd',
                       'propValuation_b', 'saleDate_b', 'Address_b',
                       'votingArea_d', 'city_b', 'postalId_b', 'sqmPrice_bd'], axis = 1)
y = finalDataLog['log_price_b']


X, y = shuffle(X, y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

X_area_train = np.array(X_train['areaWeighted_bd'])

area_normalizer = preprocessing.Normalization(input_shape=[1,], axis=None)
area_normalizer.adapt(X_area_train)

area_model = tf.keras.Sequential([
    area_normalizer,
    layers.Dense(units=1)
])

area_model.summary()

area_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

#%%time
history1 = area_model.fit(
    X_train['areaWeighted_bd'], y_train,
    epochs=200,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

hist = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('MAE [logPrice]')
  plt.legend()
  plt.grid(True)


plt.rcParams.update({'font.size': 14})
plot_loss(history1)
plt.show()

x_model1 = tf.linspace(0.0, 500, 100)
y_model1 = area_model.predict(x_model1)

def plot_area(x, y):
  plt.scatter(X_train['areaWeighted_bd'], y_train, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Area weighted')
  plt.ylabel('Log-price')
  plt.legend()


plot_area(x_model1,y_model1)
plt.show()


test_results = {}

test_results['area_model'] = area_model.evaluate(
    X_test['areaWeighted_bd'],
    y_test, verbose=0)

normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(X_train))

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
linear_model.summary()

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history2 = linear_model.fit(
    X_train, y_train,
    epochs=200,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

plot_loss(history2)
plt.show() ## AROUND 0.5 IN LOSS, ACTUALLY NOT SO BAD COMPARED TO MADSEN

test_results['linear_model'] = linear_model.evaluate(
    X_test, y_test, verbose=0)

test_results

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
      layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_area_model = build_and_compile_model(area_normalizer)
dnn_area_model.summary()

history3 = dnn_area_model.fit(
    X_train['areaWeighted_bd'], y_train,
    validation_split=0.2,
    verbose=0, epochs=200)

plot_loss(history3)
plt.show() ## RET GOD OMKRING 0.2

x_model2 = tf.linspace(0.0, 500, 251)
y_model2 = dnn_area_model.predict(x_model2)
plot_area(x_model2, y_model2)
plt.show() ## FLOT BANANFORM

test_results['dnn_area_model'] = dnn_area_model.evaluate(
    X_test['areaWeighted_bd'], y_test,
    verbose=0)

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history4 = dnn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    verbose=0, epochs=200)

plot_loss(history4)
plt.show()

test_results['dnn_model'] = dnn_model.evaluate(X_test, y_test, verbose=0)
pd.DataFrame(test_results, index=['Mean absolute error [logPrice]']).T


###################################
##### HYPER PARAMETER TUNING ######
###################################
hp_data = finalData.copy()

hp_data['log_price_b'] = np.log(hp_data['price_b'])

X_full_hp = hp_data.drop(['price_b', 'log_price_b', 'areaResidential_bd', 'AVM_price_d',
                       'quarter_b', 'quarter_numeric', 'indicatorSalesP_own',
                       'salesPeriod_b', 'energyMark_b', 'rebuildYear_bd',
                       'propValuation_b', 'saleDate_b', 'Address_b',
                       'votingArea_d', 'city_b', 'postalId_b', 'sqmPrice_bd'], axis = 1)
y_full_np = hp_data['log_price_b']

X_full_hp, y_full_np = shuffle(X_full_hp, y_full_np, random_state=0)

X_train_hp, X_test_hp, y_train_hp, y_test_hp = train_test_split(X_full_hp, y_full_np, test_size = 0.1)



## Specifying where it has to search
hp_n_neurons = hp.HParam('n_neurons', hp.Discrete([50, 60, 70, 80]))
hp_n_hiddenLayers = hp.HParam('n_hiddenLayers', hp.Discrete([2, 3, 4, 5, 5, 6, 7, 8, 9, 10]))
hp_activation = hp.HParam('activ', hp.Discrete(['relu', 'elu']))
hp_metric = 'mean_absolute_error'
hp_normalize = hp.HParam('normalize_all_layers', hp.Discrete([False, True]))
hp_DropRate = hp.HParam('DORate', hp.RealInterval(0., 0.2))
n_combo = len(hp_n_neurons.domain.values)*len(hp_n_hiddenLayers.domain.values)*len(hp_activation.domain.values)

## Defining the callbacks
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,
                              patience = 10, min_lr = 0.0001)
es_cb = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 25, verbose = 0, restore_best_weights = True)

## Preparing the log
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
    hparams=[hp_n_neurons, hp_n_hiddenLayers, hp_DropRate, hp_activation, hp_normalize],
    metrics=[hp.Metric(hp_metric, display_name='MAE')],
  )

## Actual model test

def train_test_model(hparams):
    lr = 0.005
    if hparams[hp_DropRate] > 0. :
        lr = 0.1
    X_temp, X_val, y_temp, y_val = train_test_split(X_train_hp, y_train_hp, test_size = 0.2)
    model = buildRegDnn(n_neurons = hparams[hp_n_neurons], n_hiddenLayers=hparams[hp_n_hiddenLayers],
                        DORate = hparams[hp_DropRate], activ = hparams[hp_activation],
                        normalize_all_layers = hparams[hp_normalize], learning_rate = lr, dataset=X_temp)

    model.fit(X_temp, y_temp,
            validation_split=0.1,
            verbose=0,
            epochs=400
            ,callbacks = [reduce_lr, es_cb])
    return model.evaluate(x = X_val, y = y_val, verbose = 0)

## More logging
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        MAE = train_test_model(hparams)
        print(MAE)
        tf.summary.scalar(hp_metric, MAE, step=1)

root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

#test_results
## The actual looping through values
session_num = 1

for num_units in hp_n_neurons.domain.values:
    for num_hidden in hp_n_hiddenLayers.domain.values:
        for dropRate in list(np.linspace(hp_DropRate.domain.min_value, hp_DropRate.domain.max_value, 3)):
            for activation in hp_activation.domain.values:
                for normalize in hp_normalize.domain.values:
                    hparams = {
                                hp_n_neurons: num_units,
                                hp_n_hiddenLayers: num_hidden,
                                hp_DropRate: dropRate,
                                hp_activation: activation,
                                hp_normalize: normalize
                               }
                    run_name = "run-{}".format(session_num)
                    print('--- Starting trial: {}/{}'.format(run_name, n_combo))
                    print({h.name: hparams[h] for h in hparams})
                    run('logs/hparam_tuning/' + run_name, hparams) # comment out to avoid running the very slow grid search
                    print('{}% done'.format(round(session_num/n_combo * 100, 2)))
                    session_num += 1




def buildRegDnn(n_hiddenLayers = 1, n_neurons = 30,
                activ = 'relu', init = "he_normal", lossFunc = 'mean_absolute_error',
                learning_rate = 0.05, dataset = X_train_hp, normalize_all_layers = False,
                DORate = 0.2):
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(dataset))
    model = keras.Sequential()
    model.add(normalizer)


    #adding the hidden layers.
    for i in range(n_hiddenLayers):
        if (normalize_all_layers & (i < n_hiddenLayers)):
            model.add(keras.layers.Dropout(rate = DORate))
            model.add(keras.layers.Dense(n_neurons, kernel_initializer=init))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation(activ))
        else:
            model.add(keras.layers.Dropout(rate = DORate))
            model.add(keras.layers.Dense(n_neurons, activation = activ, kernel_initializer=init))

    #adding output layer.
    model.add(layers.Dense(1))

    #compiling the model with the specified loss function and learning rate.
    model.compile(loss= lossFunc, optimizer=tf.keras.optimizers.Adam(learning_rate))

    return model
