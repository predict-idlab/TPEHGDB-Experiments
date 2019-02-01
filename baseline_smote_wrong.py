import sys

from util import *
from algorithms import classifiers

from terminaltables import AsciiTable

data = load_data()

# Defining target, categorical and continuous columns
target = 'Premature'
categorical = ['Hypertension', 'Diabetes', 'Placental_position', 
               'Bleeding_first_trimester', 'Bleeding_second_trimester', 
               'Funneling', 'Smoker']
continuous = ['Rectime', 'Age', 'Parity', 'Abortions', 'Weight', 'RMS_1', 'RMS_2', 
              'RMS_3', 'Fmed_1', 'Fmed_2', 'Fmed_3', 'Fpeak_1', 'Fpeak_2', 'Fpeak_3', 
              'Samp. en._1', 'Samp. en._2', 'Samp. en._3']

# Fit models on entire dataset
table_data = [['Algorithm', 'Sensitivity', 'Specificity', 'AUC']]

for name, algorithm in classifiers:
    print('Fitting {}...'.format(name))
    sys.stdout.flush()
    results = fit_model(algorithm, data, continuous, categorical, target, oversample_correct=False, oversample_wrong=True, ALPHA=0)
    table_data.append(
        [
            name, 
            '{}+-{}'.format(np.around(results['Sensitivity'][0], 4), 
                            np.around(results['Sensitivity'][1], 2)),
            '{}+-{}'.format(np.around(results['Specificity'][0], 4), 
                            np.around(results['Specificity'][1], 2)),
            '{}+-{}'.format(np.around(results['AUC'][0], 4), 
                            np.around(results['AUC'][1], 2)),
        ])

table = AsciiTable(table_data, 'Baseline metrics')
print(table.table)