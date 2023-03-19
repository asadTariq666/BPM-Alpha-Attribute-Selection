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
import pandas as pd
import numpy as np
### Filtered data = Complete traces - Ending in Release

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



# Path of XES File
path_of_file = 'datasets/Sepsis Cases - Event Log-Filtered.xes'

# Case Identifier
case_id = ['case:concept:name']

# Activity Label
activity_column = ['concept:name']

# Numerical Case Attributes
numerical_case_attributes = ['Leucocytes','Age','CRP','LacticAcid']

# Categorical Case Attributes
categorical_case_attributes = ['Diagnose']

# Boolean Case Attributes
boolean_case_attrbites = ['InfectionSuspected', 'DiagnosticBlood',
       'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate',
       'Infusion', 'DiagnosticArtAstrup', 'DiagnosticIC',
       'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther',
       'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature',
       'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie',
       'DiagnosticLacticAcid', 'Hypoxie',
       'DiagnosticUrinarySediment', 'DiagnosticECG']




# Importing a XES event log
from pm4py.objects.log.importer.xes import importer as xes_importer
log = xes_importer.apply(path_of_file)
#log[1],type(log)

data = pm4py.convert_to_dataframe(log)

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
# Getting mean of all numerical values
df= df.groupby(case_id, as_index=False, sort=False)[numerical_case_attributes].mean()
descriptive_features = df[numerical_case_attributes]
descriptive= descriptive_features.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(descriptive)
descriptive_features_normailsed = pd.DataFrame(x_scaled)
df[numerical_case_attributes] = descriptive_features_normailsed
df
df2 = pd.DataFrame()
# Case id
df2[case_id] = data[case_id]
#Getting categorical Features
df2[categorical_case_attributes] = data[categorical_case_attributes]
df2 = df2.groupby(case_id,as_index=False,)[categorical_case_attributes].first()
# Encoding Categorical features in to nominal categorical features
df2[categorical_case_attributes] = df2[categorical_case_attributes].apply(lambda x: pd.factorize(x)[0])
df2.head()

df3 = pd.DataFrame()
# Case id
df3[case_id] = data[case_id]
#Getting boolean Features
df3[boolean_case_attrbites] = data[boolean_case_attrbites]
df3 = df3.groupby(case_id,as_index=False,)[boolean_case_attrbites].first()
df3.replace({False: 0, True: 1}, inplace=True)
df3.head()

processed_data = df.merge(df2, how='inner', on=case_id)
processed_data = processed_data.merge(df3, how='inner', on=case_id)
processed_data = processed_data.merge(activity_count,how='inner', on=case_id)
processed_data
### Dropping nan values from columns other than diagnosis
processed_data.isna().sum()
processed_data.dropna(inplace=True)
processed_data.isna().sum()
processed_data
descriptive_features = processed_data.iloc[:,1:-1]
target = processed_data[activity_column]

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# fit the model to the training set

random_forest_classifier.fit(descriptive_features, target)

# Predict on the test set results

y_pred_100 = random_forest_classifier.predict(descriptive_features)

# Check accuracy score 

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(target, y_pred_100)))
# view the feature scores

feature_scores = pd.Series(random_forest_classifier.feature_importances_, index=descriptive_features.columns).sort_values(ascending=False)
feature_scores

feature_scores.nlargest(10).plot(kind='bar',figsize=(10,7))
plt.xticks(rotation=30)
plt.title("Important Features in Classification")
plt.show()
kms = Kmeans_Custom_RF(
	n_clusters=15,
	ordered_features=descriptive_features.columns.tolist(), 
	method='wcss_min',
).fit(descriptive_features.values)
kms.important_features[0][:10] # Features here are words
features_kms = pd.DataFrame(kms.important_features[0][0:] )
features_kms.set_index(0,inplace=True)
features_kms


features_kms.plot.bar(figsize=(20,10))
plt.xticks(rotation=30)
plt.title("Important Features in Clustering")
plt.show()