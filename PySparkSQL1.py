#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


import plotly.graph_objs as go
import plotly.plotly as py


# In[7]:


# select all categories and sort them 
from pyspark.sql.functions import ltrim, split, explode
category = yelp_business.select('categories')
individual_category = category.select(explode(split('categories', ',[ ]|,')).alias('category'))
grouped_category = individual_category.groupby('category').count()
top_category = grouped_category.sort('count',ascending=False)
# top_category.show(10,truncate=False)


# In[ ]:





# In[8]:


# select all the reviews with location and count number
business_locations = yelp_business.filter("state = \'NV\'").select('business_id', 'latitude', 'longitude')
reviewId_businessId = yelp_review.select('review_id', 'business_id')
merge_businessId = business_locations.join(reviewId_businessId, 'business_id', 'inner')
count_businessId = merge_businessId.groupby('business_id', 'latitude', 'longitude').count()
# count_businessId.sort('count', ascending = False).show()
# pandas_df = count_businessId.toPandas()


# In[9]:


# Select every business all checkin numbers
businessId_checkin_count = yelp_checkin.select('business_id',explode(split('date', ',[ ]|,')).alias('new_date')).groupby('business_id').count()
# businessId_checkin_count.sort('count', ascending = False).show(10)


# In[10]:


review_by_business = yelp_business.filter("city = \'Las Vegas\' and stars>4 and is_open=1 and review_count <500 and review_count > 200").select('categories','name', 'review_count','city', 'state', 'stars')
review_by_business.sort('review_count', ascending = False).show(10000)
review_by_business.toPandas().to_csv("../tmp/VegasSmallSpecializedPlace.csv")
#activeUser_locations = activeUser_businessId.join(business_locations, 'business_id', 'inner').select('latitude', 'longitude').groupby('latitude', 'longitude').count()

