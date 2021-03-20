# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 19:49:43 2020

@author: himit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict
import seaborn as sns
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
nltk.download('punkt')

sentences = pd.read_excel(r"C:\Users\himit\Desktop\Fall 2020\Data Mining\Project\Amazon_Manufactures_GenCategories.xlsx", index_col= 0)
sentences.head()



sid = SentimentIntensityAnalyzer()

# Array to hold sentiment
sentiments = []

# Declare variables for sentiments
compound_list = []
positive_list = []
negative_list = []
neutral_list = []




from nltk import sentiment
from nltk import word_tokenize



sentences.isnull().sum()

sentences_up=sentences.dropna(subset=['customer_reviews_substring'])
    
bodies = sentences_up['customer_reviews_substring'].to_list()
names = sentences_up['manufacturer'].to_list()
categories = sentences_up['general category'].to_list()
 
sentences_up.isnull().sum()
   
for index,body in enumerate(bodies):
    #body = bodies ['body']
    compound = sid.polarity_scores(body)['compound']
    pos = sid.polarity_scores(body)['pos']
    neg = sid.polarity_scores(body)['neg']
    neu = sid.polarity_scores(body)['neu']
        
    x=({"Name": names[index],"Category": categories[index], "Body":body,"Compound": compound,
                       "Positive": pos,
                       "Negative": neu,
                       "Neutral": neg})
    sentiments.append(x.copy())
    
print(sentiments)   
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd.to_csv("Results_manufactures.csv")
sentiments_pd.head()

# ploting the results

sent = sentiments_pd.pivot_table(index = 'Name', values ='Compound', aggfunc = np.mean)
sent

# Bar Graph

colors = ["pink","green","red","blue","yellow"]
x_axis = np.arange(len(sent.index.values))
tick_locations = [value+0.4 for value in x_axis]
fig,ax=plt.subplots(figsize=(40, 14))
plt.xticks(tick_locations, sent.index.values, rotation="horizontal")

plot=plt.bar(sent.index.values, sent["Compound"], color=colors, alpha=1, align="edge")
plt.grid()


plt.title("Overall Manufacturer sentiments based on customer text review" )
plt.xlabel("Manufacturer")
plt.ylabel("Overall rating")
ax.grid(linestyle="dotted")

plt.show()


