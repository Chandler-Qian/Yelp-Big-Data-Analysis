#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('BigDataProject').getOrCreate()

# import review data
yelp_review = spark.read.json('hdfs:///user/yq791/yelp/yelp_academic_dataset_review.json')
# import yelp business data
yelp_business = spark.read.json('hdfs:///user/yq791/yelp/yelp_academic_dataset_business.json')


# In[2]:


from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql import *

sc = spark.sparkContext
sqlContext = SQLContext(sc)


# In[3]:


import urllib, json, requests

def co2fips(lat,lon):
    try:
        url = urllib.request.urlopen("https://geo.fcc.gov/api/census/area?lat="+str(lat)+"&lon="+str(lon)+"&format=json")
    except urllib.error.HTTPError as e:
         return "0"
    except urllib.error.URLError as e:
         return "0"
    data = json.loads(url.read().decode())
    if 'results' in data and data['results']:
        county_fips=(data['results'][0]['county_fips'])
        return county_fips
    else:
        return "0"
    


# In[4]:


from pyspark.sql.functions import lit
business_locations = yelp_business.select('business_id', 'latitude', 'longitude')
reviewId_businessId = yelp_review.select('review_id', 'business_id')
merge_businessId = business_locations.join(reviewId_businessId, 'business_id', 'inner')
count_businessId = merge_businessId.groupby('business_id', 'latitude', 'longitude').count()
count_businessId.sort('count', ascending = False)
df = count_businessId


# In[5]:


from pyspark.sql.functions import udf
from pyspark.sql.types import *
#from pyspark.sql.functions import rand
udf_fips = udf(co2fips)
df=df.withColumn("fips", udf_fips(df.latitude,df.longitude))


# In[6]:


group=df.select('fips', 'count')


# In[7]:


sums = group.groupBy('fips').sum('count')


# In[11]:


sums.toPandas().to_json("../tmp/fips_all.json")


# In[15]:


import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import requests
import plotly
plotly.tools.set_credentials_file(username='ch2420', api_key='hjgqZItc1Qzj47shHxNC')


# In[250]:


pd = sums.select('fips').toPandas()
values = pd['fips']


# In[251]:


fips = values.tolist()


# In[252]:


values = (sums.select('sum(count)').toPandas())['sum(count)'].values.tolist()


# In[253]:


endpts = (np.linspace(1, 12, len(colorscale) - 1)).tolist()


# In[255]:



colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]



fig = ff.create_choropleth(
    fips=fips, values=values,
    binning_endpoints=endpts,
    colorscale=colorscale,
    show_state_data=True,
    show_hover=True, centroid_marker={'opacity': 0},
    asp=2.9, title='USA Heapmap',
    legend_title='BigData'
)
py.iplot(fig, filename='choropleth_full_usa')


# In[6]:


scl = [0.0, 'rgb(165,0,38)'], [0.3, 'rgb(215,48,39)'], [0.4, 'rgb(244,109,67)'], [0.5, 'rgb(253,174,97)'], [0.6, 'rgb(254,224,144)'], [0.7, 'rgb(224,243,248)'], [0.75, 'rgb(171,217,233)'], [0.8, 'rgb(116,173,209)'], [0.9, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']

data = [ dict(
    lat = pandas_df['latitude'],
    lon = pandas_df['longitude'],
    text = pandas_df['count'].astype(str),
    marker = dict( 
        color = pandas_df['count'],
        colorscale = scl, 
        reversescale = True,
        opacity = 0.7,
        size = pandas_df['count']/2,
        colorbar = dict(    
            thickness = 10,   
            ticklen = 10, 
            showticksuffix = "last",
            ticksuffix = " Active Users' Reviews",  

        ),
    ),
    type = 'scattergeo'
) ]

layout = dict( 
    geo = dict(
        scope = 'north america',
        showland = True,
        landcolor = "rgb(250, 250, 250)",
        subunitcolor = "rgb(100, 50, 0)",
        countrycolor = "rgb(100, 50, 0)",
        
        showlakes = True,
        lakecolor = "rgb(135,206,250)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
        projection = dict(
            type = 'conic conformal',
            rotation = dict(
                lon = -100
            )
        ),
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ -116.0, -114.0 ],
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [ 36.0, 37.0 ],
        )
    ),
    title = 'Las Vegas Active Users\' Radius',
)
fig = { 'data':data, 'layout':layout }
py.iplot(fig, filename='precipitation')


# In[ ]:




