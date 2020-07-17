#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv(r"C:\Users\HP Pavilion\OneDrive\Documents\projects\movie_metadata.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.describe(include='O')


# In[9]:


df.dropna().info()


# In[10]:


df.movie_title.duplicated()


# In[11]:


df_copy=df.copy()


# In[12]:


df_copy.drop_duplicates(['movie_title'],inplace=True)


# In[13]:


df_copy.info()


# In[14]:


df_copy.describe(include='O')


# In[15]:


df_copy.drop('movie_imdb_link',axis=1)


# In[16]:


df.duration.plot()


# In[17]:


plt.scatter(df['duration'],df['budget']/1000000000)
plt.xlabel('duration')
plt.ylabel('budget')


# In[18]:


import seaborn as sns


# In[19]:


df_copy.describe()


# In[20]:


corr=df_copy.corr()
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corr,annot=True,linewidths=0.5,square=True,cmap='Blues',ax=ax)


# In[21]:


df_copy=df_copy.drop(['facenumber_in_poster','title_year','aspect_ratio','movie_imdb_link'],axis=1)


# In[22]:


df_copy.info()


# In[23]:


corr=df_copy.corr()
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corr,annot=True,linewidths=0.5,square=True,cmap='Blues',ax=ax)


# In[24]:


df_copy['color']=df['color']


# In[25]:


df_copy.iloc[4900:4917,0:3]


# In[26]:


df_copy.info()


# In[27]:


df_copy=df_copy.reset_index(drop=True)


# In[28]:


df_copy.info()


# In[29]:


df_copy.groupby(['color']).count()


# In[30]:


df_copy['color']=df_copy['color'].map({"Color":1,"Black and  White":2,np.nan:3})


# In[31]:


df_copy.iloc[4900:4905,0:3]


# In[32]:


df_copy['color'].fillna(3,inplace=True)


# In[33]:


df_copy.iloc[4900:4905,0:3]


# In[34]:


df_copy.groupby(['color']).count()


# In[35]:


df_copy['color']=df_copy['color'].astype(int)


# In[36]:


df_copy.groupby(['color']).count()


# In[41]:


df_copy.rename(columns={"director_facebook_likes":"director_fl","num_critic_for_reviews":"num_critics","actor_3_facebook_likes":"actor3_fl","actor_1_facebook_likes":"actor1_fl","actor_2_facebook_likes":"actor2_fl","movie_facebook_likes":"movie_fl","num_user_for_reviews":"num_users"},inplace=True)


# In[42]:


df_copy.head()


# In[43]:


df_copy['num_critics'].plot()


# In[54]:


df_copy['num_critics'].hist(bins=50,color='k')


# In[59]:


df_copy['gross'].hist(bins=100,color='k')


# In[86]:


x=df_copy['budget']/10000000
x.hist(bins=95,color='k')


# In[98]:


fig,ax=plt.subplots(figsize=(20,20))
df_copy['genres'].value_counts().plot(ax=ax)


# In[100]:


len(df_copy)


# In[103]:


df_copy.shape


# In[160]:


df_copy.iloc[2,3]


# In[161]:


x=df_copy.isnull().sum(axis=1).tolist()
for index,i in enumerate(x):
    if i>=8:
        print((index,i))
        df_copy=df_copy.drop(index).reset_index(drop=True)
    
        


# In[162]:


df_copy.info()


# In[164]:


df_copy['content_rating'].value_counts().plot(kind='bar')


# In[167]:


df_copy['content_rating'].fillna(0,inplace=True)
content_mapping={"R":1,"PG-13":2,"PG":3,"Not Rated":4,"G":5,"Unrated":6,"Approved":7,"TV-14":8,"TV-MA":9,"TV-PG":10,"X":11,"TV-G":12,"Passed":13,"NC-17":14,"GP":15,"M":16,"TV-Y7":17,"TV-Y":18}
df_copy['content_rating']=df_copy['content_rating'].map(content_mapping)


# In[171]:


df_copy.groupby(['content_rating']).count()


# In[173]:


df_copy['content_rating'].fillna(0,inplace=True)


# In[175]:


df_copy.groupby(['content_rating']).count()


# In[176]:


df_copy['content_rating'].astype(int)


# In[188]:


ax=plt.subplots(figsize=(10,10))
plt.scatter(x=df_copy['content_rating'],y=df_copy['imdb_score'])


# In[193]:


df_copy[['imdb_score','content_rating']].groupby(['content_rating']).mean().sort_values('imdb_score',ascending=False)


# In[192]:


df_copy.hist(bins=30,figsize=(15,15),color='g')


# In[194]:


df_copy.columns


# In[195]:


#creating a different dataframe to analyse the most affecting features
df_a=df_copy[['num_critics','duration','num_voted_users','num_users','director_fl']]


# In[196]:


df_a.head(2)


# In[197]:


df_a.info()


# In[205]:


#filling na values
df_a["num_critics"].fillna(df_copy["num_critics"].mean(),inplace=True)
df_a["duration"].fillna(df_copy["duration"].mean(),inplace=True)
df_a["director_fl"].fillna(df_copy["director_fl"].mean(),inplace=True)
df_a["num_users"].fillna(df_copy["num_users"].mean(),inplace=True)
df_a["num_voted_users"].fillna(df_copy["num_voted_users"].mean(),inplace=True)


# In[218]:


df_a.columns


# In[228]:


df_a.info()


# In[229]:


import sklearn


# In[230]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_a,df_copy['imdb_score'])


# In[231]:


x_train.head(2)


# In[232]:


y_train.head(2)


# In[233]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
reg=LinearRegression().fit(x_train,y_train)


# In[236]:


prec_lrm=reg.predict(x_test)


# In[237]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print('The mean squared error using Linear regression is: ',mean_squared_error(y_test,prec_lrm))
print('The mean absolute error using Linear regression is: ',mean_absolute_error(y_test,prec_lrm))


# In[239]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train,y_train)


# In[241]:


prec_rf=rf.predict(x_test)


# In[242]:


print('The mean squared error using Random Forest model is: ',mean_squared_error(y_test,prec_rf))
print('The mean absolute error using Random Forest model is: ',mean_absolute_error(y_test,prec_rf))


# In[ ]:




