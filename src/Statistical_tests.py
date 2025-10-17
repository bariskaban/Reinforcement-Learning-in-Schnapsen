
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

regular_learning_gains = np.array([0.14, 0.35, -0.46, -0.05, 0.09]) 
schnapsen24_learning_gains = np.array([0.77, 0.17, 0.20, 1.02, -0.21])  


stat, p_value = ttest_rel(regular_learning_gains, schnapsen24_learning_gains)
print(f"Paired t-test: stat={stat}, p-value={p_value}")

stat, p_value = wilcoxon(regular_learning_gains, schnapsen24_learning_gains)
print(f"Wilcoxon signed-rank test: stat={stat}, p-value={p_value}")










    

        

