#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pyspark to work on spark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('PySparkSQL').getOrCreate()

# import review data
yelp_review = spark.read.json('hdfs:///user/yq791/yelp/yelp_academic_dataset_review.json')

# import yelp business data
yelp_business = spark.read.json('hdfs:///user/yq791/yelp/yelp_academic_dataset_business.json')

# import yelp user data
yelp_user = spark.read.json('hdfs:///user/yq791/yelp/yelp_academic_dataset_user.json')

# import check_in data
yelp_checkin = spark.read.json('hdfs:///user/yq791/yelp/yelp_academic_dataset_checkin.json')


# In[2]:


# select all categories and sort them 
from pyspark.sql.functions import ltrim, split, explode
category = yelp_business.select('categories')
individual_category = category.select(explode(split('categories', ',[ ]|,')).alias('category'))
grouped_category = individual_category.groupby('category').count()
top_category = grouped_category.sort('count',ascending=False)
# top_category.show(10,truncate=False)


# In[3]:


# select all the reviews with location and count number
business_locations = yelp_business.select('business_id', 'latitude', 'longitude')
reviewId_businessId = yelp_review.select('review_id', 'business_id')
merge_businessId = business_locations.join(reviewId_businessId, 'business_id', 'inner')
count_businessId = merge_businessId.groupby('business_id', 'latitude', 'longitude').count()
# count_businessId.sort('count', ascending = False).show()
# pandas_df = count_businessId.toPandas()


# In[4]:


# Select every business all checkin numbers
businessId_checkin_count = yelp_checkin.select('business_id',explode(split('date', ',[ ]|,')).alias('new_date')).groupby('business_id').count()
# businessId_checkin_count.sort('count', ascending = False).show(10)


# In[5]:


# Count every user's review amount
userId_review_amount = yelp_review.groupby('user_id').count()
userId_bigger_1000 = userId_review_amount.filter("count > 1000")
userId_bigger_1000.show()
# userId_review_amount.sort('count', ascending = False).limit(10)
active_userId = userId_bigger_1000.select('user_id')
activeUser_businessId = yelp_review.join(active_userId, 'user_id', 'inner').select('business_id')
# activeUser_business.show(10)
activeUser_locations = activeUser_businessId.join(business_locations, 'business_id', 'inner').select('latitude', 'longitude').groupby('latitude', 'longitude').count()
# activeUser_locations.sort('count', ascending = False).show(10)


# In[6]:


# choose LasVegas as the center city to find the category of Japanese
LV_Japanese_Location = yelp_business.filter((yelp_business.categories.contains('Japanese')) & (yelp_business.city == 'Las Vegas')).select('latitude', 'longitude')
LV_Japanese_Location.show()


# In[7]:





# In[ ]:





# In[4]:


# generate word cloud according to review words
from wordcloud import WordCloud
import matplotlib.pyplot as plt

review_text = yelp_review.select('text')
individual_category = review_text.select(explode(split('text', "(([^a-zA-Z]+')|('[^a-zA-Z]+))|([^a-zA-Z']+)")).alias('word'))
grouped_text = individual_category.groupby('word').count().sort('count', ascending = False)
grouped_text.show(10)
grouped_text_list = (individual_category.select('word').toPandas())['word'].values.tolist()
text_str = ' '.join(grouped_text_list)

# use nltk package to filt the adj. word
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
data = word_tokenize(text_str)
tags = set(['JJ', 'NN', 'NNS'])
processed_data_tags = nltk.pos_tag(data)
tmp_result = []
for word,pos in processed_data_tags:
    if (pos in tags):
        tmp_result.append(word)
final_result = ' '.join(tmp_result)

wordcloud = WordCloud(background_color = "white", width = 1000, height = 860, margin = 2).generate(final_result)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# save the result picture
wordcloud.to_file('test.png')


# 

# In[43]:


# Analasis percentage of Bar in several cities
business_bar = yelp_business.filter((yelp_business.categories.contains('Bar'))).count()
number_of_bar_in_every_state = yelp_business.filter((yelp_business.categories.contains('Bar'))).groupby('state').count()
number_of_business_in_every_state = yelp_business.groupby('state').count()
# number_of_bar_in_every_state.sort('count', ascending = False).withColumn('percentage', number_of_bar_in_every_state['count'] * 100 / business_bar).show(1000)
number_of_bar_in_every_state.sort('count', ascending = False).show(1000)


# In[48]:


# show bar's bar graph
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='nyuzzh', api_key='m5XdJ6Zk84xumpdyS9NL')

number_of_bar_in_every_state_pandas = number_of_bar_in_every_state.sort('count', ascending = False).toPandas()
# print(number_of_bar_in_every_state_pandas)

states, counts = [],[]
states_small, counts_small = [],[]
for index, row in number_of_bar_in_every_state_pandas.iterrows():
    state, count = row['state'], int(row['count'])
    if count >= 100:
        states.append(state)
        counts.append(count)
    else:
        states_small.append(state)
        counts_small.append(count)
states.append('Other')
counts.append(sum(counts_small))

bar_graph_data = [
    go.Bar(
        x = states,
        y = counts
    )
]
bar_count_layout = go.Layout(
    title='Distribution of Bars',
)
bar_count_figuration = go.Figure(data = bar_graph_data, layout = bar_count_layout)
py.iplot(bar_count_figuration, filename = 'bar_count')


# In[ ]:




