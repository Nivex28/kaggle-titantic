import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


train_data = pd.read_csv(r'C:\Users\Kevin\Code\Data\titanic\train.csv')
test_data = pd.read_csv(r'C:\Users\Kevin\Code\Data\titanic\test.csv')
total_data = train_data + test_data

df_all_corr = total_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

numerical_cols = ['Age', 'Fare']
categorical_cols = ['Sex', 'Pclass', 'IsAlone', 'Embarked']

X_train = train_data.copy()
y_train = train_data.Survived
X_test = test_data.copy()

X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1
X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch'] + 1
X_train['IsAlone'] = 0
X_test['IsAlone'] = 0
X_train.loc[X_train['FamilySize'] == 1, 'IsAlone'] = 1
X_test.loc[X_test['FamilySize'] == 1, 'IsAlone'] = 1
X_train = X_train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp', 'FamilySize'], axis=1)
X_test = X_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp', 'FamilySize'], axis=1)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Bundle preprocessing and modeling code in a pipeline
log_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LogisticRegression(random_state=0, max_iter=1000))])
fr_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', RandomForestClassifier(random_state=0, n_estimators=100, max_depth=None))])
svc_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LinearSVC(random_state=0, max_iter=1000, dual=False))])

# Preprocessing of training data, fit model
log_pipe.fit(X_train, y_train)
fr_pipe.fit(X_train, y_train)
svc_pipe.fit(X_train, y_train)

# Preprocessing of test data, get predictions
log_preds = log_pipe.predict(X_test)
forest_preds = fr_pipe.predict(X_test)
svc_preds = svc_pipe.predict(X_test)


# SCORES
'''
def get_score(n_estimators):
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestClassifier(n_estimators, max_depth=None, random_state=0))])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

results = {}
for i in range(1, 9):
    results[20 * i] = get_score(20 * i)
plt.plot(list(results.keys()), list(results.values()))
'''
r_sq = cross_val_score(fr_pipe, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross Validation average:{r_sq.mean()}')


'''
# OUTPUT TO CSV
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': forest_preds})
output.to_csv('my_submission.csv', index=False)
print('CSV created')
# forest pipeline way too overfitted...?, log pred decent
'''
