import numpy as np
import pandas as pd
from apyori import apriori

# Load the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Prepare the data
transactions = []
for i in range(len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1]) if str(dataset.values[i, j]) != 'nan'])

# Apply the apriori algorithm
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Convert the rules into a list
results = list(rules)

# Helper function to display results in a readable format
def inspect(results):
    lhs = []
    rhs = []
    supports = []
    
    for result in results:
        for ordered_stat in result.ordered_statistics:
            lhs.append(tuple(ordered_stat.items_base))
            rhs.append(tuple(ordered_stat.items_add))
            supports.append(result.support)
    
    return list(zip(lhs, rhs, supports))

# Display the results in a readable format
output = inspect(results)
for item in output:
    print(item)
