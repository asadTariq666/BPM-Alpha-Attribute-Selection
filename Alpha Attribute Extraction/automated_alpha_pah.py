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
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Wrapper of Random Forest classifier on K means
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


### Filtered data = Complete traces - Ending in Release
path_of_file = 'datasets/pah.xes'
case_id = ['case:concept:name']
activity_column = ['concept:name']


numerical_case_attributes = ['case:Time on Ramp']
categorical_case_attributes = ['case:Consultation Type',
       'case:Departure Delay Reason Desc', 
       'case:Team', 
       'case:Mode of Arrival Desc',
       'case:Admit Cons. Dr Specialty Desc',
       'case:Primary Diagnosis Description',
       'case:Admit Cons. Dr Primary Type Desc', 
       'case:Departure Status Code', 
       'case:Departure Referred To Code',
        'case:Type of Visit Desc',
       'case:Location after Triage', 
       'case:Mode of Departure Code',
       'case:Presenting Problem Code',
       'case:Hospital Unit',
       'case:Departure Destination Desc', 
       'case:Referred by Code', 
       'case:Admit Cons. Dr Primary Type Code',
       'case:Transfer Destn Hospital Code',
       'case:Primary Diagnosis Snomed Code', 
       'case:Mode of Arrival Code',
       'case:Admit Cons. Dr Specialty Code', 
       'case:Admission Specialty',
       'case:Type of Visit code',
       'case:Departure Status Desc', 
       'case:Presenting Problem Desc',
       'case:Referred by Desc', 
       'case:Departure Delay Reason Code', 
       'case:ATS']
boolean_case_attrbites = []
# Original event log

# Importing a XES event log
from pm4py.objects.log.importer.xes import importer as xes_importer
log = xes_importer.apply(path_of_file)
#log[1],type(log)
log[0]

data = pm4py.convert_to_dataframe(log)
data.columns
activity_df = pd.DataFrame()
# Case id
activity_df[case_id] = data[case_id]
#Getting Activity count of each case
activity_df[activity_column] = data[activity_column]
activity_count= activity_df.groupby(case_id)[activity_column].count()
activity_count.head()

df = pd.DataFrame()
# Case id
df[case_id] = data[case_id]
#Getting Numerical Features
df[numerical_case_attributes] = data[numerical_case_attributes]
df = df[~df[case_id].duplicated(keep='first')]
df = df.reset_index(drop=True)
descriptive_features = df[numerical_case_attributes]
descriptive= descriptive_features.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(descriptive)
descriptive_features_normailsed = pd.DataFrame(x_scaled)
df[numerical_case_attributes] = descriptive_features_normailsed


df2 = pd.DataFrame()
# Case id
df2[case_id] = data[case_id]
#Getting categorical Features
df2[categorical_case_attributes] = data[categorical_case_attributes]
df2 = df2.groupby(case_id,as_index=False,)[categorical_case_attributes].first()
# Encoding Categorical features in to nominal categorical features
df2[categorical_case_attributes] = df2[categorical_case_attributes].apply(lambda x: pd.factorize(x)[0])



categorical_case_attributes
df3 = pd.DataFrame()
# Case id
df3[case_id] = data[case_id]
#Getting boolean Features
df3[boolean_case_attrbites] = data[boolean_case_attrbites]
df3 = df3.groupby(case_id,as_index=False,)[boolean_case_attrbites].first()
df3.replace({False: 0, True: 1}, inplace=True)


processed_data = df.merge(df2, how='inner', on=case_id)
processed_data = processed_data.merge(df3, how='inner', on=case_id)
processed_data = processed_data.merge(activity_count,how='inner', on=case_id)

### Dropping nan values from columns other than diagnosis
processed_data.isna().sum()
processed_data.dropna(inplace=True)
processed_data.isna().sum()



descriptive_features = processed_data.iloc[:,1:-1]
target = processed_data[activity_column]

random_forest_classifier = RandomForestClassifier(n_estimators=300, random_state=42)

# fit the model to the training set

random_forest_classifier.fit(descriptive_features, target)

# Predict on the test set results

y_pred_100 = random_forest_classifier.predict(descriptive_features)

# Check accuracy score 

#print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(target, y_pred_100)))
# view the feature scores

feature_scores = pd.Series(random_forest_classifier.feature_importances_, index=descriptive_features.columns).sort_values(ascending=False)
print("Important features in Classification")
print(feature_scores.nlargest(20))

kms = Kmeans_Custom_RF(
	n_clusters=13,
	ordered_features=descriptive_features.columns.tolist(), 
	method='wcss_min',
).fit(descriptive_features.values)
kms.important_features[0][:10] # Features here are words
features_kms = pd.DataFrame(kms.important_features[0][0:20] )
features_kms.set_index(0,inplace=True)
print("Important features in Clustering")
print(features_kms)


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,15))
features_kms.plot.barh( ax=ax1,xlabel='Attributes',ylabel="Relative Importance",title="Clustering")
feature_scores.nlargest(20).plot(kind='barh', ax=ax2,xlabel='Attributes',ylabel="Relative Importance",title="Classification")
plt.xlabel("Relative Importance")
plt.show()