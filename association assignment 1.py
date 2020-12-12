# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
!pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# Importing the dataset
book= pd.read_csv("book.csv")
book.shape
book.describe()

book.isna().sum()          # Checking for any NA values
# No NA values in the dataset

# Value counts for each book
counts= book.apply(pd.Series.value_counts)

# Bar plot of counts for each book
ax= plt.subplots(figsize=(12,5))
plt.bar(
        x= book.columns,
        height= counts.iloc[1,],
        width= 0.8,
        color= "rgbymc");
plt.xlabel("counts");plt.ylabel("items")
# Frequency of items for CookBks is the highest with 862 counts and lowest for ItalAtlas with 74 counts

#Apriori algorithm
# Using the apriori alogorithm and filtering the books with minimun support of 1%
frequent_itemsets= apriori(book, min_support=0.01, max_len=3, use_colnames= True)
frequent_itemsets.shape

# Sorting the frequent items in descending order
frequent_itemsets.sort_values('support', ascending= False, inplace= True)

# Since there are many itemsets with 1% support, let's change the minimum support to 10%
frequent_itemsets2= apriori(book, min_support=0.1, max_len=3, use_colnames=True)
frequent_itemsets2.head()
frequent_itemsets2.shape

# Sorting the frequent items in descending order
frequent_itemsets2.sort_values('support', ascending= False, inplace=True)
frequent_itemsets2.sort_values
frequent_itemsets2.shape
# we have got total 39 frequent itemsets with 10% support

#Association rule
# Applying association rule with minimum threshold value of lift as 1
rules= association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules.head()
rules.shape
# Filtering the rules again with minimum confidence of 50% and minimun lift of 2

rules= rules[ (rules['lift']>= 2) &
             (rules['confidence']>=0.5) ]

# Sorting the rules by lift in descending order
rules= rules.sort_values('lift', ascending= False)
rules.shape
# Finally have got total of 17 rules with minimum support of 10% , minimum confidence of 50% and minimum lift of 1
# Showing top 5 rules
rules.head(5)

#Visualizations
# importing necessary libraries
import random            
import matplotlib.pyplot as plt
# Defining the support and confidence from finalized rules to plot them

support= rules['support'].values
confidence= rules['confidence'].values

# plot of support v/s confidence
plt.scatter(support, confidence, alpha=0.5, marker="o")
plt.xlabel("support")
plt.ylabel("confidence")
# All the rules have confidence in the range of 0.5 to 0.7 except for one rule which has the highest confidence of 100% and support of 11%
# plot of support v/s lift

plt.scatter(rules['support'], rules['lift'], alpha= 0.8)
plt.xlabel("support")
plt.ylabel("lift")
# Rules with higher support has low lift values and rules with lower support has higher lift
# plot of confidence v/s lift

plt.scatter(rules['confidence'], rules['lift'], alpha= 1)
plt.xlabel("support")
plt.ylabel("lift")
# All the rules have confidence in the range of 0.5 to 0.7 except for one rule which has the highest confidence of 100% and highest lift of 2.32