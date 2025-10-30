import pandas as pd

from apriori import Apriori

# Groceries dataset (small sample)
data = {
    'TransactionID': [1,1,1,2,2,3,3,3,4,4,5,5],
    'Item': ['milk','bread','butter','bread','jam','milk','bread','eggs','bread','butter','milk','bread']
}

df = pd.DataFrame(data)
transactions = df.groupby('TransactionID')['Item'].apply(list).tolist()

print("Sample transactions:")
for t in transactions:
    print(t)


ap = Apriori(min_support=0.4, min_confidence=0.6)
ap.fit(transactions)

print("\nFrequent Itemsets:")
for k, v in ap.frequent_itemsets.items():
    print(f"{k}-itemsets:", v)

print("\nAssociation Rules:")
print(ap.get_rules())