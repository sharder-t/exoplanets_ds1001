"""
NYU Center for Data Science
DS-GA 1001 Fall 2020
Term Project
Authors: Sharder Islam, and 3 others
"""

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, plot_roc_curve
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

# Input data and prepare for modeling
path = r"cumulative.csv"
df = pd.read_csv(path)
df.drop(df[list(df.filter(regex='err'))], axis=1, inplace=True)
df.drop(columns=["koi_tce_delivname", "koi_tce_plnt_num", "koi_time0bk"], inplace=True)
df.columns = df.columns.str.replace('koi_', '')
df.insert(0, 'Y_binary', np.where(df["disposition"] == "FALSE POSITIVE", 0, 1), False) #1 = not false positive, 0 = false positive
df = df.loc[df["disposition"] != "CANDIDATE",:]

# Split and prepare feature and target columns
first_feature_idx = df.columns.get_loc("period") # target variable index
X = df.iloc[:, first_feature_idx:] # all validated column features after koi_score (based on index)
X.fillna(X.mean(), inplace=True)  # Impute missing values with mean
y = df["Y_binary"]
X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75)

# Fit base classifiers and generate predicted columns
dt = DecisionTreeClassifier().fit(X_train, y_train)
lr = LogisticRegression().fit(X_train, y_train)
SVM = SVC(kernel="linear").fit(X_train, y_train)

dt_pred = dt.predict(X_test)
LR_pred = lr.predict(X_test)
SVM_pred = SVM.predict(X_test)

# Plot base classifiers
fig, ax = plt.subplots()
plot_roc_curve(lr, X_test, y_test, ax=ax)
plot_roc_curve(dt, X_test, y_test, ax=ax)
plot_roc_curve(SVM, X_test, y_test, ax=ax)
plt.savefig("base.png", dpi=500)

# Feature importances based off Decision Tree model
imps = pd.DataFrame({"importance": dt.feature_importances_,
                     "feat": df.columns[first_feature_idx:]}).sort_values(by="importance", ascending=False)

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_ylim(ymin=0, ymax=0.30)
ax.set_title("Feature importances from Decision Tree Classifier")
ax.set_xticklabels(imps.feat)
imps.plot.bar(rot=60, width=0.75, ax=ax)

for i, v in enumerate(imps.importance):
    ax.text(i-0.25, v+0.005, str(round(v, 3)), rotation=60, fontweight=36)


# --- Grid Search ------------------------------------------------------------ #
#pipeline parameters
parameters = [
        {'clf': [LogisticRegression()],
            'clf__penalty' : ['l1', 'l2'],
            'clf__C' : np.logspace(-4, 4, 20),
            'clf__solver' : ['lbfgs', 'liblinear']},
        {'clf': [SVC()],
            'clf__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            'clf__C' : [10**i for i in np.linspace(-3, 3, 7)],
            'clf__degree' : np.linspace(1, 3, 3)},
        {'clf': [DecisionTreeClassifier()],
            'clf__min_samples_split' : np.linspace(0.005, 0.05, 10),
            'clf__min_samples_leaf' : np.linspace(0.005, 0.05, 10)}]

# evaluating multiple classifiers based on pipeline parameters
# -------------------------------
result=[]

for params in parameters:

    #classifier
    clf = params['clf'][0]

    #getting arguments by popping out classifier
    params.pop('clf')

    #pipeline
    steps = [('clf',clf)]

    #cross validation using Grid Search
    grid = GridSearchCV(Pipeline(steps), param_grid=params, n_jobs=-1, verbose=20, cv=5)
    best = grid.fit(X_test, y_test)

    # storing result
    result.append\
    ({'grid': grid,
      'classifier': grid.best_estimator_,
      'best score': grid.best_score_,
      'best params': grid.best_params_,
      'cv': grid.cv})

# # Best model from grid search (best.best_estimator_)
best_model = DecisionTreeClassifier(min_samples_leaf=0.005, min_samples_split=0.015)
plot_roc_curve(best_model, X_test, y_test)


# --- Ensemble classifiers --------------------------------------------------- #
rf = RandomForestClassifier().fit(X_train, y_train)
ada = AdaBoostClassifier().fit(X_train, y_train)
gbc = GradientBoostingClassifier().fit(X_train, y_train)

# Plot AUC for ensemble classifiers
fig, ax = plt.subplots()
plot_roc_curve(rf, X_test, y_test, ax=ax)
plot_roc_curve(ada, X_test, y_test, ax=ax)
plot_roc_curve(gbc, X_test, y_test, ax=ax)
plt.savefig("ensemble.png", dpi=500)

# Create SHAP explainers for best model
model = RandomForestClassifier().fit(X_train, y_train)
shap_values = shap.TreeExplainer(model).shap_values(X_train)

shap.summary_plot(shap_values[0], X_train, feature_names=X.columns, plot_type="violin")
shap.summary_plot(shap_values[0], X_train, feature_names=X.columns, plot_type="bar")
