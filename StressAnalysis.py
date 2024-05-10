#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing libraries
import nltk
nltk.download('punkt')
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


# In[3]:


df = pd.read_csv('Reddit.csv')
df


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df['subreddit'].nunique()


# In[8]:


dfunique = df['subreddit'].unique() 

dfunique


# In[9]:


df['id'].nunique()


# In[10]:


newDF = df.columns[9:113]
df.drop(columns=newDF, inplace=True, axis=1)
df


# In[11]:


df['Word Count'] = df['text'].apply(lambda x: len(x.split()))
df


# In[12]:


# mapped each subreddit to have a number as a representation of it, its the column on the far right
df['subreddit_id'] = df['subreddit'].map({'relationships' : 0, 'anxiety' : 1, 'ptsd' : 2, 'assistance': 3,
                                        'homeless' : 4, 'almosthomeless' : 5, 'domesticviolence': 6, 
                                          'survivorsofabuse' : 7, 'stress' : 8, 'food_pantry': 9, })
df


# In[13]:


df['RealTime'] = pd.to_datetime(df['social_timestamp'], unit='s')
df
df['hour'] = df['RealTime'].dt.hour.astype(int)
df['year'] = df['RealTime'].dt.year.astype(int)
df['month'] = df['RealTime'].dt.month.astype(int)
df
df_clean = df[df['social_karma'] <= 1000].reset_index(drop=True)
df = df_clean


# In[ ]:


from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
scores1 = df['text'].iloc[0]
analyzer.polarity_scores(scores1)
compoundList = []
for index, row, in  df.iterrows():
   text = row.text
   scores = analyzer.polarity_scores(scores1)
   compound = scores['compound']
   print(format(index,'2d'), format(compound, '6.2f'), row.text)
   compoundList.append(compound)


# In[15]:


def compoundScore(text):
    scores1 = analyzer.polarity_scores(text)
    return scores1['compound']


df['compound'] = df['text'].apply(compoundScore)
df


# In[16]:


group = df.groupby('subreddit_id')['compound'].transform('mean')
df['MeanCompoundScore'] = group


# In[17]:


X = df[['social_num_comments']]
Y = df[['social_karma']]


# In[18]:


# setting linear regression varaibles
# randomstate 103 gives .60
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=103)


# In[19]:


# stuck here, can't get the model to fit due to error
model = LinearRegression()
model.fit(X_train, y_train)


# In[20]:


y_test


# In[21]:


X_test


# In[22]:


model.score(X_test, y_test)


# In[23]:


predictions = model.predict(X_test)
print('r_squared : ', r2_score(y_test, predictions))
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))


# In[24]:


plt.scatter(X_test, y_test, color='blue', marker='.', label='Actual')
plt.xlabel('Number of Comments')
plt.ylabel('Social Karma')
plt.title('Linear Regression Test')
plt.plot(X_test, predictions, color='red', label='Predicted')
plt.legend()
plt.show()


# In[25]:


wc = df.pivot_table(values='Word Count', index='subreddit', aggfunc='mean')
wc2 = wc.plot(kind='bar', stacked=True)


# In[26]:


df.pivot_table(values='post_id', index='subreddit', aggfunc='count')


# In[27]:


df.pivot_table(values='post_id', index='subreddit', aggfunc='count')
df2 = df.pivot_table(values='post_id', index='subreddit', aggfunc='count')
df2
df3 = df2.plot(kind='bar', stacked=True)
df3


# In[ ]:


allTextList = df.text.to_list()
allText = ' '.join(allTextList)
allText


# In[29]:


tokens = nltk.word_tokenize(allText)
tokens[:8]


# In[30]:


nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')


# In[31]:


keepwords = [word for word in tokens if word not in stopwords]
keepwords


# In[ ]:


keepwords2 = [word for word in tokens if word.isalpha()]
keepwords2


# In[ ]:


keepwords3 = [word.lower() for word in tokens if word.lower()]
keepwords3


# In[34]:


from nltk import FreqDist
frequency = nltk.FreqDist(keepwords3)
frequency.plot(30)


# In[35]:


sk = df['subreddit']
sr = df['social_karma']
group = df.groupby('subreddit')['social_karma'].mean()
plt.title("Mean Social Karma scores by each post in subreddits")
plt.bar(group.index, group.values)
plt.ylabel('Karma')
plt.xlabel('Subreddits')
plt.xticks(rotation=50, ha='center')
plt.show


# In[36]:


x = df['subreddit_id']
y = df['Word Count']
group = df.groupby('subreddit_id').size()
names = ['almosthomeless', 'anxiety', 'assistance', 'domesticviolence', 'food_pantry', 'homeless', 'ptsd', 'relationships', 'stress', 'survivorsofabuse']
plt.xticks(rotation=45)
plt.title("Total Word count by Subreddit in each post")
plt.pie(group, labels = names, autopct='%1.0f%%', radius = 1.1)
plt.figure(figsize=(19, 8))
plt.show()


# In[37]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud


# In[38]:


cloud = WordCloud(width=1920, height=1080, background_color = 'white').generate_from_frequencies(frequency)
plt.imshow(cloud, interpolation = 'bilinear')


# In[39]:


plt.title("Mean Social Karma scores by each post in subreddits")
plt.bar(group.index, group.values)
plt.ylabel('Karma')
plt.xlabel('Subreddits')
plt.xticks(rotation=50, ha='center')
plt.legend(names)
plt.show

