import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
import matplotlib.colors
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import time
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from statsmodels.api import add_constant


finalData = pd.read_csv("finalData.csv")
finalData.columns
finalData = finalData.drop(['Unnamed: 0'], axis = 1)
finalData.info()

# We modify roofType a bit
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

X = finalData.drop(['price_b', 'areaResidential_bd', 'AVM_price_d', 'quarter_b', 'quarter_numeric',
                    'indicatorSalesP_own', 'salesPeriod_b', 'energyMark_b',
                    'rebuildYear_bd', 'propValuation_b', 'saleDate_b',
                    'Address_b', 'votingArea_d', 'city_b', 'postalId_b', 'sqmPrice_bd'], axis = 1)

y = finalData['price_b']

X.info()

X, y = shuffle(X, y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
        ])

pipe = pipe.fit(X_train, y_train)

y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)



#plt.figure(figsize = (14, 6))
plt.rcParams["figure.figsize"] = (14,6)

ax = plt.subplot(1,2,1)
sns.residplot(x=y_train_pred, y=y_train_pred - y_train, lowess=False, color = 'royalblue',
                              scatter_kws={'alpha': 0.3},
                              line_kws={'color': 'darkblue', 'lw': 1, 'alpha': 0.8}, label='Training data')
sns.residplot(x=y_test_pred, y=y_test_pred - y_test, color = 'green',
                              scatter_kws={'alpha': 0.3}, lowess=False,
                              line_kws={'color': 'darkgreen', 'lw': 1, 'alpha': 0.8}, label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')

ax = plt.subplot(1,2,2)
plt.scatter(y_train, y_train_pred, alpha=0.3, c = 'royalblue')
plt.scatter(y_test, y_test_pred, alpha=0.3, c = 'green')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
plt.ylabel('Model predictions')
plt.xlabel('Truths')
plt.xlim([0, 25000000])
plt.ylim([0, 25000000])

plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/OLS_resid_plot.png')

plt.show()

###
# QQ PLOT
###

residuals1 = y_train_pred - y_train
residuals2 = y_test_pred - y_test

plt.figure(figsize = (14, 6))

ax = plt.subplot(1,2,1)

stats.probplot(residuals1, dist=norm, plot=plt, fit=True)
ax.get_lines()[0].set_markerfacecolor('steelblue')
ax.get_lines()[0].set_markeredgecolor('steelblue')
plt.title("Normal Q-Q Plot")

ax = plt.subplot(1,2,2)

stats.probplot(residuals2, dist=norm, plot=plt, fit=True)
ax.get_lines()[0].set_markerfacecolor('limegreen')
ax.get_lines()[0].set_markeredgecolor('limegreen')
plt.title("Normal Q-Q Plot")
plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/OLS_QQ_plot_1.png')
plt.show()

#####################
### LOG TRANSFORM ###
#####################

finalDataLog = finalData.copy()

finalDataLog['log_price_b']=np.log(finalDataLog['price_b']) # this is the natural log (ln)

X = finalDataLog.drop(['price_b', 'areaResidential_bd', 'log_price_b', 'AVM_price_d',
                       'quarter_b', 'quarter_numeric',
                       'indicatorSalesP_own', 'salesPeriod_b', 'energyMark_b',
                       'rebuildYear_bd', 'propValuation_b', 'saleDate_b',
                       'Address_b', 'votingArea_d', 'city_b', 'postalId_b', 'sqmPrice_bd'], axis = 1)

y = finalDataLog['log_price_b']


X, y = shuffle(X, y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
        ])

pipe = pipe.fit(X_train, y_train)
y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)


plt.figure(figsize = (14, 6))

ax = plt.subplot(1,2,1)

sns.residplot(x=y_train_pred, y=y_train_pred - y_train, lowess=False, color = 'royalblue',
                              scatter_kws={'alpha': 0.3},
                              line_kws={'color': 'darkblue', 'lw': 1, 'alpha': 0.8}, label='Training data')
sns.residplot(x=y_test_pred, y=y_test_pred - y_test, color = 'green',
                              scatter_kws={'alpha': 0.3}, lowess=False,
                              line_kws={'color': 'darkgreen', 'lw': 1, 'alpha': 0.8}, label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')

ax = plt.subplot(1,2,2)


plt.scatter(y_train, y_train_pred, alpha=0.3, c = 'royalblue')
plt.scatter(y_test, y_test_pred, alpha=0.3, c = 'green')
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
plt.ylabel('Model predictions')
plt.xlabel('Truths')
plt.xlim([12.5, 18.5])
plt.ylim([12.5, 18.5])

plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/OLS_resid_plot_log.png')

plt.show()

### QQ PLOT LOG TRANSFORM

residuals1 = y_train_pred - y_train
residuals2 = y_test_pred - y_test

plt.rcParams['figure.figsize'] = (14, 6)
ax = plt.subplot(1,2,1)

stats.probplot(residuals1, dist=norm, plot=plt, fit=True)
ax.get_lines()[0].set_markerfacecolor('steelblue')
ax.get_lines()[0].set_markeredgecolor('steelblue')
plt.title("Normal Q-Q Plot")

ax = plt.subplot(1,2,2)

stats.probplot(residuals2, dist=norm, plot=plt, fit=True)
ax.get_lines()[0].set_markerfacecolor('limegreen')
ax.get_lines()[0].set_markeredgecolor('limegreen')
plt.title("Normal Q-Q Plot")

plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/OLS_QQ_plot_log.png')

plt.show()

###################################
### MOST IMPORTANT COEFFICIENTS ###
###################################

feature_names = np.array(X.columns)
coefs = pd.DataFrame(
    pipe.named_steps['regressor'].coef_,
    columns=['Coefficients'], index=feature_names
)


imp_coef = pd.concat([coefs.sort_values(by=['Coefficients']).head(8),
                     coefs.sort_values(by=['Coefficients']).tail(8)])

color = ['#FFD500' if y<0 else '#005BBB' for y in imp_coef['Coefficients']]
fig, ax1 = plt.subplots()
ax1.barh(imp_coef.index, imp_coef['Coefficients'], color = color)
plt.title('Coefficients')
plt.yticks(rotation = 45)
plt.savefig('D:/Dropbox/Apps/Overleaf/Thesis and project - UCPH/Figures/OLS_coeff_headTail.png')
plt.show()
# blå: #005BBB, gul: #FFD500 Slava Ukraini!

coefs['Coefficients_abs'] = np.abs(coefs['Coefficients'])
coefs.sort_values(by=['Coefficients_abs'])

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_colwidth', None)

fit = sm.OLS(y_train,X_train).fit()

p_values = pd.DataFrame(fit.pvalues, columns=['p values'])
p_values.sort_values(by=['p values'], ascending=False, inplace=True)
p_values['p values'] = p_values['p values'].round(4)
p_values.head(10) # FIVE COEFS WITH P VALUE > 0.05

coefs = pd.concat([coefs, p_values], axis=1)
coefs.drop(columns=['Coefficients'], inplace=True)
print(coefs.sort_values(by=['Coefficients_abs']).head(10).to_latex(index=True))

len(coefs.loc[coefs['p values']>0.05])

##################
### ASSESSMENT ###
##################

train_sizes, train_scores, test_scores = learning_curve(pipe, X_train, y_train,
                                                        train_sizes = np.linspace(0.1, 1.0, 20),
                                                        cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Train R2')

plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Val R2')

plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('R2')
plt.legend(loc='lower right')
#plt.ylim([0.84,0.87])
plt.show()


results1 = pd.DataFrame({'MSE, train' : [(mean_squared_error(y_train, y_train_pred))],
                        'MAE, train' : (mean_absolute_error(y_train, y_train_pred)),
                        'R^2, train' : (r2_score(y_train, y_train_pred)),
                        'MSE, test' : (mean_squared_error(y_test, y_test_pred)),
                        'MAE, test' : (mean_absolute_error(y_test, y_test_pred)),
                        'R^2, test' : (r2_score(y_test, y_test_pred))},
                        index = ['OLS'])
print(results1.to_latex(index=True))

mean_percent = np.mean(abs(np.exp(y_test)-np.exp(y_test_pred))/np.exp(y_test))*100

median_percent = np.median(abs(np.exp(y_test)-np.exp(y_test_pred))/np.exp(y_test))*100

def within15(y_pred, y_test):
    dataset = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test}, columns=['y_pred', 'y_test'])
    dataset['exp_y_pred'] = np.exp(dataset.y_pred)
    dataset['exp_y_test'] = np.exp(dataset.y_test)
    dataset['difference'] = abs(dataset.exp_y_test-dataset.exp_y_pred)
    dataset['diffpercent'] = dataset.difference / dataset.exp_y_test *100
    return (len(dataset[dataset.diffpercent < 15]) / len(dataset)) *100

dev1 = pd.DataFrame({'Mean deviation %' : [mean_percent],
                    'Median deviation %' : [median_percent],
                    'Within 15 %': [within15(y_test_pred, y_test)]},
                        index = ['OLS'])

print(dev1.round(2).to_latex(index=True))
