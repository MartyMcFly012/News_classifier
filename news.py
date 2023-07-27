# pip3 install matplotlib
# pip3 install scikit-learn
# pip3 install -U imbalanced-learn

# Step zero: install the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# Step 1a. First, add a class column that is a zero for fake, and one for true
fake = pd.read_csv("Fake.csv")
fake.insert(4, "class", 0)


true = pd.read_csv("True.csv")
true.insert(4, "class", 1)

df = pd.concat([fake, true])
print(df)


# Step 1b. Next, combine the two dataframes
