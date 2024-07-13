#!/usr/bin/env python
# coding: utf-8

# # start the task of creating a job recommendation system by importing the necessary Python libraries and the dataset

# In[1]:


#Installing wordcloud
get_ipython().system('pip install wordcloud')


# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string

data = pd.read_csv("jobs.csv")
print(data.head())


# # Data Pre-Processing 
# The dataset has an unnamed column. Let’s remove it and move further

# In[3]:


data = data.drop("Unnamed: 0",axis=1)


# # Now let’s have a look if the dataset contains any null values or not

# In[4]:


data.isnull().sum()


# # As the dataset doesn’t have any null values, let’s move further by exploring the skills mentioned in the Key Skills column

# In[5]:


text = " ".join(i for i in data["Key Skills"])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # let’s have a look at the functional areas mentioned in the dataset

# In[6]:


text = " ".join(i for i in data["Functional Area"])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # Now let’s have a look at the job titles mentioned in the dataset
# 

# In[7]:


text = " ".join(i for i in data["Job Title"])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# # Creating a Content-Based Recommendation System
# 

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer

feature = data["Key Skills"].tolist()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)


# # Now I will set the Job title column as the index of the dataset so that the users can find similar jobs according to the job they are looking for

# In[9]:


indices = pd.Series(data.index, index=data['Job Title']).drop_duplicates()


# # Sample of how will the recommendation system looks

# In[10]:


def jobs_recommendation(Title, similarity = similarity):
    index = indices[Title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[::], reverse=True)
    similarity_scores = similarity_scores[0:5]
    newsindices = [i[0] for i in similarity_scores]
    return data[['Job Title', 'Job Experience Required', 
                 'Key Skills']].iloc[newsindices]

print(jobs_recommendation("Software Developer"))



# # Actual Job Recommendation System For Different Job Titles

# In[12]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import ipywidgets as widgets
from IPython.display import display

# Load the data
data = pd.read_csv('jobs.csv')

# Create TF-IDF matrix
feature = data["Key Skills"].tolist()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)

# Fit NearestNeighbors model
nn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(tfidf_matrix)

# Create a mapping of job titles to indices
indices = pd.Series(data.index, index=data['Job Title']).drop_duplicates()

def jobs_recommendation(job_title):
    if job_title not in indices:
        print(f"Job title '{job_title}' not found in the dataset.")
        return None
    
    index = indices[job_title]
    distances, indices_nn = nn.kneighbors(tfidf_matrix[index], n_neighbors=6)
    
    job_indices = indices_nn[0][1:]  # Exclude the first one as it is the same job
    recommendations = data[['Job Title', 'Job Experience Required', 'Key Skills']].iloc[job_indices]
    display(recommendations)

# Create dropdown menu for job titles
job_titles = data['Job Title'].unique().tolist()
dropdown = widgets.Dropdown(
    options=job_titles,
    description='Job Title:',
    disabled=False,
)

def on_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        jobs_recommendation(change['new'])

dropdown.observe(on_change)

# Display the dropdown menu
display(dropdown)


# In[ ]:




