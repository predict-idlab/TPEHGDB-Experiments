from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# Defining all algorithms and their grids for hyper-parameter tuning
classifiers = [
    (
        'Logistic Regression', 
        GridSearchCV(
            LogisticRegression(random_state=2018),
            {
                'penalty': ['l1', 'l2'], 
                'C': [10**i for i in range(-5, 3)]
            }, scoring='roc_auc')
    ), (
        'Decision Tree',
        GridSearchCV(
            DecisionTreeClassifier(random_state=2018), 
            {
                'criterion': ['entropy', 'gini'], 
                'max_depth': [None, 3, 5, 10, 15],
                'max_features': [None, 'auto'],
                'min_samples_split': [2, 5, 10],
            }, scoring='roc_auc')
    ), (
        'Linear Discriminant Analysis',
        LinearDiscriminantAnalysis()
    ), (
        'Quadratic Discriminant Analysis',
        QuadraticDiscriminantAnalysis()
    ), (
        'K-Nearest Neighbors',
        GridSearchCV(
            KNeighborsClassifier(), 
            {
               'n_neighbors': [3, 5, 7, 11, 15, 21], 
               'metric': ['minkowski', 'manhattan'],
            }, scoring='roc_auc'),
    ), (
        'Random Forest',
        GridSearchCV(
            RandomForestClassifier(random_state=2018), 
            {
               'criterion': ['entropy', 'gini'], 
               'max_depth': [None, 3, 5, 10, 15],
               'max_features': [None, 'auto'],
               'n_estimators': [10, 100, 500],
            }, scoring='roc_auc'),
    ), (
        'Support Vector Machine',
        GridSearchCV(
            SVC(probability=True, random_state=2018), 
            {
                'kernel': ['linear', 'rbf'], 
                'C': [10**i for i in range(-5, 3)]
            }, scoring='roc_auc')
    )
]