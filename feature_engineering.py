import math

df = pd.read_pickle('yelp_dataframe.pkl')
catgory = {}
for i, restaurant in df.iterrows():
    for category in restaurant['categories_new']:
        alias = category['alias']
        try:
            catgory[alias] += [i]
        except:
            catgory[alias] = [i]

def distance(ilat,jlat,ilong,jlong):
    R = 6371.e3
    phi1 = math.radians(ilat)
    phi2 = math.radians(jlat)
    deltaphi = math.radians(jlat-ilat)
    deltalambda = math.radians(jlong-ilong)
    a = math.sin(deltaphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(deltalambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c / 1609.34
    return d

df['restaurant_density'] = np.nan
df['restaurant_similar_density'] = np.nan
df['zprice_all'] = np.nan
df['zreview_count_all'] = np.nan
df['zreview_per_week_all'] = np.nan
df['zstar_all'] = np.nan
df['is_chain'] = np.nan

for i, restaurant in df.iterrows():
    price_all_list = []
    review_count_all_list = []
    review_per_week_all_list = []
    star_all_list = []
    density_similar_list = []
    density_all_list = []
    
    ilong = restaurant['coordinates.longitude']
    ilat = restaurant['coordinates.latitude']
    for category in restaurant['categories_new']:
        alias = category['alias']
        for restaurant2 in catgory[alias]:
            jlong = df.loc[restaurant2]['coordinates.longitude']
            jlat = df.loc[restaurant2]['coordinates.latitude']
            dist = distance(ilat,jlat,ilong,jlong)
            if dist <= 1:
                density_similar_list += [restaurant2]
  
    df.loc[i,'restaurant_similar_density'] = len(density_similar_list)
    
    for j, restaurant2 in df.iterrows():
        jlong = restaurant2['coordinates.longitude']
        jlat = restaurant2['coordinates.latitude']
        dist = distance(ilat,jlat,ilong,jlong)
        if dist <= 1:
            price_all_list += [restaurant2['price']]
            review_count_all_list += [restaurant2['review_count']]
            review_per_week_all_list += [restaurant2['reviews_per_week']]
            star_all_list += [restaurant2['stars']]
            density_all_list += [j]
        
    df.loc[i,'zprice_all'] = (restaurant['price']-np.nanmean(price_all_list))/4.
    df.loc[i,'zreview_count_all'] = (restaurant['review_count']-np.nanmean(review_count_all_list))/np.nanstd(review_count_all_list)
    df.loc[i,'zreview_per_week_all'] = (restaurant['reviews_per_week']-np.nanmean(review_per_week_all_list))/np.nanstd(review_per_week_all_list)
    df.loc[i,'zstar_all'] = (restaurant['stars']-np.nanmean(star_all_list))/5.
        
    df.loc[i,'restaurant_density'] = len(density_all_list)
    df.loc[i,'restaurant_similar_density'] = len(density_similar_list)
    # True if there are more than one
    df.loc[i,'is_chain'] = (len(df[df['name_new'] == restaurant['name_new']]) > 1)