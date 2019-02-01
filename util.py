import pandas as pd

import numpy as np
np.random.seed(2018)  # Seeding for reproducibility

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

from tsfresh.feature_selection.relevance import calculate_relevance_table

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.disabled = True

from tqdm import tqdm

def process_header_file(file):
    # The clinical variables are added as comments in the
    # .hea file. Extract them from that file.
    start_idx = 0
    with open(file, 'r') as ifp:
        lines = ifp.readlines()
        for line_idx, line in enumerate(lines):
            if line.startswith('#'):
                start_idx = line_idx
                break
        
        names = []
        values = []
        for line in lines[start_idx+1:]:
            _, name, value = line.split()
            names.append(name)
            values.append(value)
            
        return names, values

def load_data():
    # Read the CSV
    data = pd.read_csv('data/features/tpehgdb_features__filter_0.3_Hz-3.0_Hz.fvl')
    # Name the columns
    data.columns = ['Record', 'Chann', 'Gestation', 'Rec. time', 'Group',
                    'RMS', 'Fmed', 'Fpeak', 'Samp. en.', 'Premature', 'Early']
    # Remove spaces from record identifiers
    data['Record'] = data['Record'].apply(lambda x: x.strip())

    # Remove the special sample
    data = data[data['Record'] != 'tpehg873']
    
    # Each sample has three rows (one for each channel)
    # Pivot the table to create N/3 rows and M*3 columns
    features = data.pivot(index='Record', columns='Chann', 
                          values=['RMS', 'Fmed', 'Fpeak', 'Samp. en.'])
    features = features.reset_index()

    # Rename the columns
    new_cols = []
    for col in ['RMS', 'Fmed', 'Fpeak', 'Samp. en.']:
        new_cols.extend(['{}_{}'.format(col, i) for i in range(1, 4)])
    features.columns = ['Record'] + new_cols
    
    # Merge with record ID and label
    features = features.merge(data[['Record', 'Premature']], left_on='Record', right_on='Record')
    features['Premature'] = features['Premature'].map({'f': 0, 't': 1})
    
    # Extract clinical features from .hea files
    vectors = []
    for record in set(data['Record']):
        names, values = process_header_file('data/tpehgdb/{}.hea'.format(record))
        vectors.append([record]+values)
    clinical_df = pd.DataFrame(vectors, columns=['Record'] + names)
    
    # Merge it all together
    data = features.merge(clinical_df, left_on='Record', right_on='Record')
    data = data.drop_duplicates().reset_index(drop=True)
    data = data.sort_values(by='Record')
    
    return data

def oversample_data(X, y):
    smote = SMOTE(random_state=1337)
    return smote.fit_resample(X, y)

def fit_model(clf, data, continuous, categorical, target, oversample_correct=False,
              oversample_wrong=False, ALPHA=0.05):
    # We will store our metrics in these lists
    aucs = []
    sensitivities = []
    specificties = []
    cms = []

    # Partition data into feature matrix and label vector
    X = data[continuous + categorical]
    y = data[target]

    # One-hot-encode categorical variables
    X = pd.get_dummies(X, columns=categorical, drop_first=True)

    # Different ways to indicate a value is missing, map all to NaN
    X = X.replace('None', np.NaN)
    X = X.replace('inf', np.NaN)
    X = X.replace('-inf', np.NaN)
    X = X.replace(np.inf, np.NaN)
    X = X.replace(-np.inf, np.NaN)

    cols = list(X.columns)

    # Oversample the entire dataset
    # WARNING: NEVER DO THIS! THIS DRASTICALLY BIASES THE RESULTS
    if oversample_wrong:
        X, y = oversample_data(X.fillna(X.median()), y)
        X = pd.DataFrame(X, columns=cols)
        y = pd.Series(y)
    
    # Apply stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
    for fold_nr, (train_idx, test_idx) in tqdm(enumerate(skf.split(X, y)), total=5):

        # Partition into training and testing set
        X_train = X.iloc[train_idx, :]
        X_test = X.iloc[test_idx, :]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fill NaNs with medians
        for col in continuous:
            if sum(pd.isnull(X_train[col])) + sum(pd.isnull(X_test[col])) > 0:
                X_test[col] = X_test[col].fillna(np.nanmedian(X_test[col].astype(float).values))
                X_train[col] = X_train[col].fillna(np.nanmedian(X_train[col].astype(float).values))
            
        # Apply feature selection using the training set
        if ALPHA > 0:
            rel_table = calculate_relevance_table(X_train.astype(float), y_train)
            relevant_features = list(rel_table[rel_table['p_value'] < ALPHA]['feature'])

            if relevant_features:
                X_train = X_train[relevant_features]
                X_test  = X_test[relevant_features]

        # Oversample the training data if needed
        if oversample_correct:
            X_train, y_train = oversample_data(X_train, y_train)

        # Fit a model
        clf.fit(X_train, y_train)

        # Evaluate the model
        try:
            roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            cm = confusion_matrix(y_test, clf.predict(X_test))
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (fn + tp)
            specificity = tn / (tn + fp)
            cms.append(cm)
            aucs.append(roc_auc)
            sensitivities.append(sensitivity)
            specificties.append(specificity)
        except:
            pass

        
    return {
        'CM': np.sum(cms, axis=0),
        'AUC': (np.mean(aucs), np.std(aucs)),
        'Sensitivity': (np.mean(sensitivities), np.std(sensitivities)),
        'Specificity': (np.mean(specificties), np.std(specificties))
    }
