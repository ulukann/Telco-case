
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("/Users/ulukan/PycharmProjects/pythonProject/osman/machine_learning/odev/TelcoChurn/Telco-Customer-Churn.csv")

df.head()

df.info()

df.shape

df.describe().T

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T



def grab_col_names(dataframe, cat_th=10, car_th=7):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


num_cols = num_cols + cat_but_car

num_cols = [col for col in num_cols if col not in "customerID"]


df.info()

df.head()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df.isnull().sum()

## nan_indexes = df[df['TotalCharges'].isna()].index
## df["TotalCharges"] boş değerlerin indeksleri


df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")

cat_cols_ratio = len(cat_cols) / len(df.columns) * 100
num_cols_ratio = len(num_cols) / len(df.columns) * 100

labels = ['Kategorik Sütunlar', 'Sayısal Sütunlar']
sizes = [cat_cols_ratio, num_cols_ratio]
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.title('Sütun Türü Dağılımı')
plt.show()

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)

df.info()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


df["Churn"] = df["Churn"].map({'No':0,'Yes':1})

df.head()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"Churn",col)

df.groupby("SeniorCitizen").agg({"Churn": "count"})

df.info()

df.isnull().sum()

df["Churn"].value_counts()


df.describe([0.01, 0.05,0.25, 0.75, 0.90, 0.99]).T


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

df.head()

df.isnull().sum()

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

df.loc[(df['tenure'] >= 0) & (df['tenure'] < 24), 'tenure_q'] = 'new_customer'
df.loc[(df['tenure'] >= 24) & (df['tenure'] < 48), 'tenure_q'] = "standard"
df.loc[(df['tenure'] >= 48) & (df['tenure'] <= 72), 'tenure_q'] = 'old_customer'

df.head()

df.groupby("tenure_q").agg({"Churn": ["mean","count"]})

df["gender"] = df["gender"].map({'Male':0,'Female':1})

df.loc[((df['gender'] == 0) & (df["MultipleLines"]== "Yes")), 'PHONE_LINE_GENDER'] ="multiple_lines__male"
df.loc[((df['gender'] == 0) & (df["MultipleLines"]== "No")), 'PHONE_LINE_GENDER'] ="single_line__male"
df.loc[((df['gender'] == 0) & (df["MultipleLines"]== "No phone service")), 'PHONE_LINE_GENDER'] ="no_line__male"
df.loc[((df['gender'] == 1) & (df["MultipleLines"]== "Yes")), 'PHONE_LINE_GENDER'] ="multiple_lines__female"
df.loc[((df['gender'] == 1) & (df["MultipleLines"]== "No")), 'PHONE_LINE_GENDER'] ="single_line__female"
df.loc[((df['gender'] == 1) & (df["MultipleLines"]== "No phone service")), 'PHONE_LINE_GENDER'] ="no_line__female"

df.groupby("PHONE_LINE_GENDER").agg({"Churn": ["mean","count"]})


df.loc[((df['gender'] == 0) & (df["Partner"]== "Yes")), 'PARTNER_GENDER'] ="with_partner_male"
df.loc[((df['gender'] == 0) & (df["Partner"]== "No")), 'PARTNER_GENDER'] ="without_partner_male"
df.loc[((df['gender'] == 1) & (df["Partner"]== "Yes")), 'PARTNER_GENDER'] ="with_partner_female"
df.loc[((df['gender'] == 1) & (df["Partner"]== "No")), 'PARTNER_GENDER'] ="without_partner_female"

df.groupby("PARTNER_GENDER").agg({"Churn": ["mean","count"]})

df.head()

df.info()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

le = LabelEncoder()

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

df.head()

df.info()

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.describe().T


dff = df.copy()

y = df["Churn"]
X = df.drop(["customerID", "Churn"], axis=1)


rf_model = RandomForestClassifier(random_state=22)
rf_model.get_params()


cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7904281511927789
cv_results['test_f1'].mean()
# 0.5515955408596234
cv_results['test_roc_auc'].mean()
# 0.8214982994718202


rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=22).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7988058107672471
cv_results['test_f1'].mean()
# 0.5397432129095903
cv_results['test_roc_auc'].mean()
# 0.8438525277541423


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")


gbm_model = GradientBoostingClassifier(random_state=22)

gbm_model.get_params()

cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8032091788179881
cv_results['test_f1'].mean()
# 0.5861727941835791
cv_results['test_roc_auc'].mean()
# 0.845700372108497

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=22).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.806191427672753
cv_results['test_f1'].mean()
# 0.5898368696194629
cv_results['test_roc_auc'].mean()
# 0.8477764434463702


xgboost_model = XGBClassifier(random_state=22, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7856042123685398
cv_results['test_f1'].mean()
# 0.5598527655212105
cv_results['test_roc_auc'].mean()
# 0.8241697905481207

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_best_grid.best_params_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=22).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8033511234595782
cv_results['test_f1'].mean()
# 0.5825558160470621
cv_results['test_roc_auc'].mean()
# 0.8464400953038071


lgbm_model = LGBMClassifier(random_state=22)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7966799269307698
cv_results['test_f1'].mean()
# 0.5795668742393136
cv_results['test_roc_auc'].mean()
# 0.8358838128666772

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=22).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8047722836957224
cv_results['test_f1'].mean()
# 0.5851150611698369
cv_results['test_roc_auc'].mean()
# 0.8457621640258723


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)



catboost_model = CatBoostClassifier(random_state=22, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7968204601909801
cv_results['test_f1'].mean()
# 0.5704956105994485
cv_results['test_roc_auc'].mean()
# 0.8401248579499319


catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catboost_best_grid.best_params_

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.80718554422866
cv_results['test_f1'].mean()
# 0.5860469004165128
cv_results['test_roc_auc'].mean()
# 0.848346918572643










































