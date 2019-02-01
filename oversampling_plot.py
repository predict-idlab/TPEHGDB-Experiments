import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_orig, y_orig = make_classification(n_samples=100, n_features=2, n_informative=2, class_sep=0.5,
                                     n_redundant=0, weights=[0.8, 0.2], random_state=2018)


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_orig, y_orig)
X_new, y_new = X_resampled[len(X_orig):, :], y_resampled[len(y_orig):]
print(X_new.shape, y_new.shape)
X_train_correct, X_test_correct, y_train_correct, y_test_correct = train_test_split(X_orig, y_orig, test_size=0.5, 
                                                                                    stratify=y_orig, random_state=2018)
X_train_incorrect, X_test_incorrect, y_train_incorrect, y_test_incorrect = train_test_split(X_new, y_new, test_size=0.5, 
                                                                                            stratify=y_new, 
                                                                                            random_state=2018)

plt.figure()
mscatter(X_orig[:, 0], X_orig[:, 1], 
		 c=['r' if x == 1 else 'b' for x in y_orig], 
		 m=['o' if x == 1 else 's' for x in y_orig], 
		 s=75)
plt.axis('off')
plt.savefig('figures/original.svg', format='svg')

# Correct approach
plt.figure()
mscatter(X_train_correct[:, 0], X_train_correct[:, 1], 
	     c=['r' if x == 1 else 'b' for x in y_train_correct], 
		 m=['o' if x == 1 else 's' for x in y_train_correct], 
	     s=75)
plt.axis('off')
plt.savefig('figures/correct_train.svg')

plt.figure()
mscatter(X_test_correct[:, 0], X_test_correct[:, 1], 
         c=['r' if x == 1 else 'b' for x in y_test_correct], 
	     m=['o' if x == 1 else 's' for x in y_test_correct], 
         s=75)
plt.axis('off')
plt.savefig('figures/correct_test.svg')

# Apply over-sampling correctly, on solely the training data
# Apply SMOTE
smote = SMOTE(random_state=1337)
X_train_sampled_correct, y_train_sampled_correct = smote.fit_resample(X_train_correct, y_train_correct)
plt.figure()
mscatter(X_train_correct[:, 0], X_train_correct[:, 1], 
	     c=['r' if x == 1 else 'b' for x in y_train_correct], 
		 m=['o' if x == 1 else 's' for x in y_train_correct], 
	     s=75)
mscatter(X_train_sampled_correct[:, 0], X_train_sampled_correct[:, 1], 
	     edgecolors=['r' if x == 1 else 'b' for x in y_train_sampled_correct], 
		 m=['o' if x == 1 else 's' for x in y_train_sampled_correct], 
		 facecolors='none',
	     s=75)
plt.axis('off')
plt.savefig('figures/correct_train_sampled.svg')


plt.figure()
mscatter(X_orig[:, 0], X_orig[:, 1], 
	     c=['r' if x == 1 else 'b' for x in y_orig], 
		 m=['o' if x == 1 else 's' for x in y_orig], 
	     s=75)
mscatter(X_new[:, 0], X_new[:, 1], 
	     edgecolors=['r' if x == 1 else 'b' for x in y_new],
		 m=['o' if x == 1 else 's' for x in y_new], 
		 facecolors='none',
		 s=75)
plt.axis('off')
plt.savefig('figures/incorrect_sampled.svg')

plt.figure()
mscatter(X_train_correct[:, 0], X_train_correct[:, 1], 
	     c=['r' if x == 1 else 'b' for x in y_train_correct], 
	     m=['o' if x == 1 else 's' for x in y_train_correct],
	     s=75)
mscatter(X_train_incorrect[:, 0], X_train_incorrect[:, 1], 
	     edgecolors=['r' if x == 1 else 'b' for x in y_train_incorrect], 
	     m=['o' if x == 1 else 's' for x in y_train_incorrect],
	     facecolors='none',
	     s=75)
plt.axis('off')
plt.savefig('figures/incorrect_train.svg')

plt.figure()
mscatter(X_test_correct[:, 0], X_test_correct[:, 1], 
	     c=['r' if x == 1 else 'b' for x in y_test_correct], 
	     m=['o' if x == 1 else 's' for x in y_test_correct],
	     s=75)
mscatter(X_test_incorrect[:, 0], X_test_incorrect[:, 1], 
	     edgecolors=['r' if x == 1 else 'b' for x in y_test_incorrect], 
	     m=['o' if x == 1 else 's' for x in y_test_incorrect],
	     facecolors='none',
	     s=75)
plt.axis('off')
plt.savefig('figures/incorrect_test.svg')


"""
# Apply over-sampling correctly, on solely the training data
# Apply SMOTE
smote = SMOTE(random_state=1337)
X_train_sampled_correct, y_train_sampled_correct = smote.fit_resample(X_train_correct, y_train_correct)

# Filter out the old points from X_train_sampled
X_train_new, y_train_new = [], []
for i in range(len(X_train_sampled_correct)):
    record = tuple(X_train_sampled_correct[i, :])
    new = True
    for j in range(len(X_train_correct)):
        if record == tuple(X_train_correct[j, :]):
            new = False
            break
    if new:
        X_train_new.append(X_train_sampled_correct[i, :])
        y_train_new.append(y_train_sampled_correct[i])
        
X_train_new_correct = np.array(X_train_new)
y_train_new_correct = np.array(y_train_new)

# Plot data
plt.figure()
plt.scatter(X_train_new_correct[:, 0], X_train_new_correct[:, 1], 
            c=['g' if x == 1 else 'b' for x in y_train_new_correct], marker='x', s=75)
plt.scatter(X_train_correct[:, 0], X_train_correct[:, 1], 
            c=['r' if x == 1 else 'b' for x in y_train_correct], s=75)
plt.scatter(X_test_correct[:, 0], X_test_correct[:, 1], 
            c=['r' if x == 1 else 'b' for x in y_test_correct], alpha=0.25, s=75)
#plt.title('Applying over-sampling after data partitioning')
plt.axis('off')
plt.savefig('figures/final_result_correct.svg')

plt.figure()
plt.scatter(X_train_correct[:, 0], X_train_correct[:, 1], c=['r' if x == 1 else 'b' for x in y_train_correct], s=75)
plt.scatter(X_test_correct[:, 0], X_test_correct[:, 1], c=['r' if x == 1 else 'b' for x in y_test_correct], alpha=0.2, s=75)
plt.scatter(X_train_incorrect[:, 0], X_train_incorrect[:, 1], c=['g' if x == 1 else 'b' for x in y_train_new], marker='x', s=75)
plt.scatter(X_test_incorrect[:, 0], X_test_incorrect[:, 1], c=['g' if x == 1 else 'b' for x in y_train_new], marker='x', alpha=0.2, s=75)
#plt.title('Applying over-sampling before data partitioning')
plt.axis('off')
plt.savefig('figures/final_result_incorrect.svg')
"""