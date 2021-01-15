# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
!pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# Importing the dataset
mov= pd.read_csv("my_movies.csv")
mov.shape
mov.columns

# Removing the first 5 columns as they have been converted to dummy variables
mov= mov.iloc[:,5:]

mov.isna().sum()          # Checking for any NA values
# No NA values in the dataset

# Value counts for each movie
counts= mov.apply(pd.Series.value_counts)
counts

# Bar plot of counts for each movie
ax= plt.subplots(figsize=(12,5))
plt.bar(
        x= mov.columns,
        height= counts.iloc[1,],
        width= 0.8,
        color= "rgbkymc");
plt.xlabel("counts"); plt.ylabel("items")
# Frequency of items for "Gladiator" movie is the highest with 7 counts and lowest for 3 other films just 1 count each

#Apriori algorithm
# Using the apriori alogorithm and filtering the movies with minimun support of 1%
frequent_itemsets= apriori(mov, min_support=0.01, max_len=3, use_colnames= True)
frequent_itemsets.shape
frequent_itemsets.head()

# Sorting the frequent items in descending order
frequent_itemsets.sort_values('support', ascending= False, inplace= True)

# Since there are many itemsets with 1% support, let's change the minimum support to 10%
frequent_itemsets2= apriori(mov, min_support=0.1, max_len=3, use_colnames= True)

frequent_itemsets2.sort_values('support', ascending= False, inplace= True)
frequent_itemsets2.shape

# Since there are many itemsets with 10% support, let's change the minimum support to 20%
frequent_itemsets3= apriori(mov, min_support=0.2, max_len=3, use_colnames= True)

# Sorting the frequent items in descending order
frequent_itemsets3.sort_values('support', ascending= False, inplace= True)
frequent_itemsets3.shape
# we have got total 13 frequent itemsets with 20% support

#Association rules
# Applying association rule with minimum threshold value of lift as 1
rules= association_rules(frequent_itemsets3, metric="lift", min_threshold=1)
rules.head()
rules.shape
# Filtering the rules again with minimum confidence of 50% and minimun lift of 1

rules= rules[ (rules['lift']>=1) &
              (rules['confidence']>=0.5)]
rules.shape

# Sorting the rules by lift in descending order
rules= rules.sort_values('support', ascending= False).head(10)
rules
# Finally have got total of 15 rules with minimum support of 10% , minimum confidence of 50% and minimum lift of 1
# Showing top 5 rules
rules.head()

# Observations
# Rule 1 shows that people who have seen "LOTR2" movie will also watch "LOTR1" movie with 100% confidence and lift of 5.
# Since both the movies belong to same category. Viewer will be obviously interested in the second movie of same category

#Visualizations
# importing necessary libraries
import random

# Defining the support and confidence from finalized rules to plot them
support= rules['support'].values
confidence= rules['confidence'].values

# plot of support v/s confidence
plt.scatter(support, confidence, alpha=0.5, marker="o")
plt.xlabel("support")
plt.ylabel("confidence")
plt.show()
# All the rules have support in the range of 0.4 to 0.6 except for two rules which has the highest confidence of 100% and support of 20%

# plot of support v/s lift
plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()
# All the rules have lift in the range of 1 to 1.5 except two rules which have lift of 5 with support of 20%.

# plot of confidence v/s lift
plt.scatter(rules['confidence'], rules['lift'], alpha=0.5)
plt.xlabel('confidence')
plt.ylabel('lift')
plt.title('Confidence vs Lift')
plt.show()
# All the rules have confidence in the range of 0.5 to 0.9 except for two rules which has the highest confidence of 100% and lift of 5