# Find Alpha Attribute from Sepsis dataset. clustering/classification
### Reading data
import os
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
import warnings
warnings.simplefilter("ignore")
from pm4py.objects.log.log import EventLog
# from ClusterFlags import bag_of_activities
from pm4py.objects.log.util import get_log_representation as get_
from sklearn.decomposition import PCA
from pm4py.objects.log.obj import EventLog, Trace
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
### Filtered data = Complete traces - Ending in Release
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


sepsis = pd.read_csv('/Users/asadtariq/Downloads/BPM-Alpha-Attribute-Selection/Datasets/Sepsis Cases - Event Log -Filtered.csv')
sepsis.head()
### Getting Release Type of every case
rel = sepsis[['Case ID', 'Activity']]
release = rel[rel['Activity'].str.contains('Release')]
### Counting number of activities
count_events = sepsis[['Case ID','Activity','Leucocytes', 'CRP', 'LacticAcid']]
count_events_and_tests= count_events.groupby('Case ID')[['Activity','Leucocytes', 'CRP', 'LacticAcid']].count()
count_events_and_tests.head()

### Dropping non important features.
sepsis = sepsis.drop(['Activity', 'Complete Timestamp', 'Variant', 'Variant index',
    'lifecycle:transition', 'org:group'], axis=1)
sepsis.head(20)
### Dropping rows with NAN values i.e. keeping 1 row for each case
sep=sepsis.dropna(thresh=15)
sep.reset_index(drop=True, inplace=True)
sep.shape
len(sepsis.Diagnose.unique())

count_events_and_tests =count_events_and_tests.reset_index()
### Dropping Numeric tests counts
count_events_and_tests
# Renaming Column names of dataframe of Counts
count_events_and_tests.rename(columns={'Activity': 'Activity_count', 'Leucocytes': 'Leucocytes_count','CRP':'CRP_count','LacticAcid':'LacticAcid_count'}, inplace=True)
count_events_and_tests = count_events_and_tests[['Case ID','Activity_count']]
release.rename(columns={'Activity': 'Release_type'}, inplace=True)
### Dropping Tests from sepsis dataset
sep = sep.drop(['Leucocytes', 'CRP', 'LacticAcid'], axis=1)
sep.head()
### Merging All dataframes to constitute table with additional columns i.e. Count of activities and Tests conducted
mer =pd.merge(count_events_and_tests, release, on='Case ID', how='outer')
mer.shape
mer
processed_data = pd.merge(sep, mer, on='Case ID', how='outer')
processed_data.head(10)
### Data Cleaning
processed_data.isna().sum()
# Deleting Case with null case id, Dropping rows with Age and Diagnosis null.
df = processed_data[processed_data['Case ID'].notna()]
df = df[df['Diagnose'].notna()]
df.isna().sum()
df.head()
print(df.describe())
# Converting True and False in to 1 and 0
df.replace({False: 0, True: 1}, inplace=True)
df.head()
df.select_dtypes(include=['object'])
# Encoding Diagnose and Release Type categorical features
df[['Diagnose', 'Release_type']] = df[['Diagnose', 'Release_type']].apply(lambda x: pd.factorize(x)[0])
df.head()
df.shape
df.to_csv('Datasets/Processed_cleaned_data.csv')
# Clustering 
## Clustering on Descriptive Features
descriptive_features = df.drop(['Case ID'],axis=1)
descriptive_features.head()
##  Elbow Method to extract optimal number of clusters


within_cluster_sum_of_squares = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    kmeans.fit(descriptive_features)
    within_cluster_sum_of_squares.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 10), within_cluster_sum_of_squares)

plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()
## Keeping filtered cases in the main event log

el = pd.read_csv('Datasets/Sepsis Cases - Event Log.csv')
cases = df['Case ID']
el = el.loc[el['Case ID'].isin(cases)] 
el.shape

dataframe = pm4py.format_dataframe(el, case_id='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')
event_log = pm4py.convert_to_event_log(dataframe)


kmeans = KMeans(
        init="random",
        n_clusters=2,
        n_init=10,
        max_iter=300,
        random_state=42
    )
km = kmeans.fit(descriptive_features)



already_seen = {}
labels = km.labels_
clusters = []

for i in range(len(event_log)):
        if not labels[i] in already_seen:
            already_seen[labels[i]] = len(list(already_seen.keys()))
            clusters.append(EventLog())
        trace = event_log[i]
        clusters[already_seen[labels[i]]].append(trace)
## Clustering  on Normailsed Descriptive Features
### Normalise dataframe

from sklearn import preprocessing

x = descriptive_features.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
descriptive_features_normailsed = pd.DataFrame(x_scaled)
descriptive_features_normailsed.head()

kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=10,
        max_iter=300,
        random_state=42
    )
km = kmeans.fit(descriptive_features_normailsed)



already_seen = {}
labels = km.labels_
clusters = []

for i in range(len(event_log)):
        if not labels[i] in already_seen:
            already_seen[labels[i]] = len(list(already_seen.keys()))
            clusters.append(EventLog())
        trace = event_log[i]
        clusters[already_seen[labels[i]]].append(trace)
### Finding Important Feature of K means Clustering
class Kmeans_Custom_RF(KMeans):
    def __init__(self, ordered_features, method='wcss_min', **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.ordered_features = ordered_features

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        
        if self.method == 'unsup2sup':
            self.important_features = self.get_importance_unsup2sup(X)
        
        elif self.method == 'wcss_min':
            self.important_features = self.get_importance_wcss_min()
        return self

    def get_importance_wcss_min(self):
        labels = self.n_clusters
        centroids = np.abs(self.cluster_centers_)
        sorted_centroid_features_idx = np.argsort(centroids, axis=1)[:,::-1]

        weighted_features = {}
        for label, centroid in enumerate(sorted_centroid_features_idx):
            ordered_weights = centroids[label][centroid]
            ordered_features = [self.ordered_features[feature] for feature in centroid]
            weighted_features[label] = list(zip(ordered_features, ordered_weights))
        
        return weighted_features

    def get_importance_unsup2sup(self, X):

        weighted_features = {}
        for label in range(self.n_clusters):
            binary_encoding = np.array([1 if x == label else 0 for x in self.labels_])
            classifier = RandomForestClassifier()
            classifier.fit(X, binary_encoding)

            sorted_features_idx = np.argsort(classifier.feature_importances_)[::-1]
            ordered_features = np.array(self.ordered_features)[sorted_features_idx]
            ordered_weights = classifier.feature_importances_[sorted_features_idx]
            weighted_features[label] = list(zip(ordered_features, ordered_weights))
        return weighted_features
descriptive_features = descriptive_features.loc[:, descriptive_features.columns != "Unnamed: 0"]


kms = Kmeans_Custom_RF(
	n_clusters=2,
	ordered_features=descriptive_features.columns.tolist(), 
	method='wcss_min',
).fit(descriptive_features.values)
kms.important_features[0][:10] # Features here are words
features_kms = pd.DataFrame(kms.important_features[0][0:10] )
features_kms.set_index(0,inplace=True)

features_kms.plot.bar(figsize=(10,5))
plt.xticks(rotation=30)
plt.title("ID-K Imporant Features")
plt.show()
### Target feature = Activity_count
target = descriptive_features['Activity_count']
descriptive = descriptive_features.loc[:, descriptive_features.columns != "Activity_count"]
descriptive = descriptive.loc[:, descriptive.columns != "Unnamed: 0"]

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# fit the model to the training set

random_forest_classifier.fit(descriptive, target)

# Predict on the test set results

y_pred_100 = random_forest_classifier.predict(descriptive)

# Check accuracy score 

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(target, y_pred_100)))
# view the feature scores

feature_scores = pd.Series(random_forest_classifier.feature_importances_, index=descriptive.columns).sort_values(ascending=False)
feature_scores
type(feature_scores)
feature_scores.nlargest(5).plot(kind='bar',figsize=(10,7))
plt.xticks(rotation=45)
plt.ylabel("Relative Importance")
plt.xlabel("Features")
plt.title("ID-R Imporant Features")
plt.show()