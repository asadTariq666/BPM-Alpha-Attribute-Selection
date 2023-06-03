### This script adds new alpha attributes to the event log 
## Importing librarires
import pm4py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pm4py.objects.log.util import get_log_representation as get_
warnings.simplefilter("ignore")
from sklearn.decomposition import PCA
from pm4py.objects.log.obj import EventLog, Trace
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pm4py.objects.log.log import EventLog
import os
from pm4py.objects.conversion.log import converter as log_converter
import gzip
import shutil
def optimal_k(X, max_clusters):
    wcss = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    k = np.argmin(diff_r) + 2
    return k
## Importing log
# file and path of Event log
path_of_file = 'Datasets/Sepsis Cases - Event Log-Filtered.xes' 
case_id = 'case:concept:name'
# Activities label in Event log
activity = 'concept:name' 
# Redundant activities i.e. Activities existing more than once in a trace
activities_array = []

# Setting parameters for clustering method

n_init_km=10
max_iter_km=500
random_state_km=42
# Importing a XES event log
from pm4py.objects.log.importer.xes import importer as xes_importer
log = xes_importer.apply(path_of_file)
### Extracting total Activities
from pm4py.algo.filtering.log.attributes import attributes_filter
activities = attributes_filter.get_attribute_values(log, activity)
activities

### Converting activity keys in to a list
### Converting activity keys in to a list
activities_array = activities.keys()
activities_array = list(activities_array)


# ## Converting log in to a pandas dataframe
dataframe2 = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

##  Creating Set of Activities - Extracting Activities from the Event log
set_vector, features = get_.get_representation(log, str_ev_attr=[activity], str_tr_attr=[],
                                                  num_ev_attr=[], num_tr_attr=[], str_evsucc_attr=[])

### sublogs on merged vector
merged = set_vector

# Example usage:
X = set_vector
optimal_clusters = optimal_k(X, max_clusters=20)
print("Optimal number of clusters:", optimal_clusters)

 # Setting parameters for number of sublogs
number_of_sublogs=optimal_clusters
kmeans = KMeans(
        init="k-means++",
        n_clusters=number_of_sublogs,
        n_init=n_init_km,
        max_iter=max_iter_km,
        random_state=random_state_km
    )
km = kmeans.fit(merged)

already_seen = {}
labels = km.labels_
clusters = []

for i in range(len(log)):
        if not labels[i] in already_seen:
            already_seen[labels[i]] = len(list(already_seen.keys()))
            #clusters.append(EventLog())
        trace = log[i]
        clusters.append(labels[i])


## PCA - Dimension Reduction. 
pca = PCA(n_components=3)
pca.fit(merged)
data3d = pca.transform(merged)
   
### sublogs on merged vector
kmeans2 = KMeans(
        init="k-means++",
        n_clusters=number_of_sublogs,
        n_init=n_init_km,
        max_iter=max_iter_km,
        random_state=random_state_km
    )
km2 = kmeans2.fit(data3d)

already_seen2 = {}
labels2 = km2.labels_
clusters2 = []

for i in range(len(log)):
        if not labels2[i] in already_seen2:
            already_seen2[labels2[i]] = len(list(already_seen2.keys()))
            #clusters2.append(EventLog())
        trace = log[i]
        clusters2.append(labels2[i])



### CLustering on Normalised Data 
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(merged)
descriptive_features_normailsed = pd.DataFrame(x_scaled)
descriptive_features_normailsed.head()
kmeans3 = KMeans(
        init="k-means++",
        n_clusters=number_of_sublogs,
        n_init=n_init_km,
        max_iter=max_iter_km,
        random_state=random_state_km
    )
km3 = kmeans3.fit(descriptive_features_normailsed)

already_seen3 = {}
labels3 = km3.labels_
clusters3 = []

for i in range(len(log)):
        if not labels3[i] in already_seen3:
            already_seen3[labels3[i]] = len(list(already_seen3.keys()))
            #clusters3.append(EventLog())
        
        clusters3.append(labels3[i])

cluster_df = pd.DataFrame()
# Case id
cluster_df[case_id] = dataframe2[case_id].unique()
cluster_df

cluster_df["Clustering Simple"] = clusters
cluster_df["Clustering PCA"] = clusters2
cluster_df["Clustering Normalised"] = clusters3

processed_data = dataframe2.merge(cluster_df, how='inner', on=case_id)
duplicate_indices = processed_data[processed_data['case:concept:name'].duplicated()].index

silhouette_avg = silhouette_score(merged, kmeans.labels_)
silhouette_avg2 = silhouette_score(merged, kmeans2.labels_)
silhouette_avg3 = silhouette_score(merged, kmeans3.labels_)
silhouette_avg,silhouette_avg2,silhouette_avg3

processed_data.loc[duplicate_indices, 'Clustering Simple'] = np.nan
processed_data.loc[duplicate_indices, 'Clustering PCA'] = np.nan
processed_data.loc[duplicate_indices, 'Clustering Normalised'] = np.nan

dataframe = pm4py.format_dataframe(processed_data)
log = log_converter.apply(dataframe,
variant=log_converter.Variants.TO_EVENT_LOG,
parameters={"stream_postprocessing":True})
pm4py.write_xes(log, "postprocessed.xes")
filename = "postprocessed.xes"

with open(filename, 'rb') as f_in:
    with gzip.open(filename + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

values = [silhouette_avg,silhouette_avg2,silhouette_avg3]
print('Silhouette average scores for each clustering method are :', values)


# Set the labels for the x-axis (optional)
labels = ['Cluster Simple', 'Cluster PCA', 'Cluster Normalised']
# Create a bar plot
plt.bar(range(len(values)), values)

# Set the x-tick labels to the labels we defined earlier
plt.xticks(range(len(values)), labels)

# Add a title and axis labels
plt.title('Comparison of clustering method.')
plt.xlabel('Method')
plt.ylabel('Score')

# Set the x-tick labels to the labels we defined earlier
plt.xticks(range(len(values)), labels)

# Display the plot
plt.show()