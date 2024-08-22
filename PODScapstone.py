#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 01:23:31 2024

@author: alanwu
"""

#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve
import seaborn as sns
import random

#RANDOM SEED FROM MY N-NUMBER
N = 11262055
np.random.seed(N)
random.seed(N)

#load dataset
data = pd.read_csv('spotify52kData.csv')
summary = data.describe()
#print(summary)

#data cleaning
missing_data = data.isnull().sum() #no missing data found in dataset
#drop duplicate songs
columns_remove = ['album_name', 'songNumber']
duplicates = data[data.duplicated(data.columns.difference(columns_remove), keep = False )]
data = data.drop_duplicates(data.columns.difference(columns_remove), keep='first')


#QUESTION 1
print("Q1")
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

#Create histograms for each feature
plt.figure(figsize=(20, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 5, i)
    sns.histplot(data[feature], kde=False)
    plt.title(feature)

plt.tight_layout()
plt.show()

#determine if normal, using the Shapiro-Wilk test
from scipy.stats import shapiro

# Perform Shapiro-Wilk test and collect results
normality_results = {'Feature': [], 'Statistic': [], 'p-value': [], 'Normally Distributed': []}
for feature in features:
    stat, p = shapiro(data[feature])
    normality_results['Feature'].append(feature)
    normality_results['Statistic'].append(stat)
    normality_results['p-value'].append(f'{p:.6f}')
    normality_results['Normally Distributed'].append('Yes' if p > 0.05 else 'No')


normality_df = pd.DataFrame(normality_results)

# Display the DataFrame
print(normality_df)

#QUESTION 2
print("Q2")

#scatterplot of duration vs popularity
plt.figure(figsize=(10, 6))
data['duration_seconds'] = data['duration']/1000 #convert duration into seconds
plt.scatter(data['duration_seconds'], data['popularity'], alpha=0.5, color='gray')
plt.title('Scatterplot of Song Duration vs. Popularity')
plt.xlabel('Duration (seconds)')
plt.ylabel('Popularity')
plt.show()

#Calculate the correlation coefficient (R)
correlation_coefficient, p_value = pearsonr(data['duration'], data['popularity'])
print(f'Correlation coefficient: {correlation_coefficient:.6f}')
print(f'P-value: {p_value:.6f}')

#QUESTION 3
print("Q3")
#Split data into two groups, nonexplicit and explicit songs
explicit_songs = data[data['explicit'] == True]['popularity']
non_explicit_songs = data[data['explicit'] == False]['popularity']

print("Explicit median:",explicit_songs.median())
print("NonEx median:", non_explicit_songs.median())

# Perform the Mann-Whitney U test
u_stat, p_value = mannwhitneyu(explicit_songs, non_explicit_songs, alternative = 'greater')

# Print the results
print(f'U-statistic: {u_stat}')
print(f'P-value: {p_value}')

#Histogram for nonexplicit songs
plt.hist(non_explicit_songs, bins=30, alpha=0.5, label='Non-Explicit', color='blue')
#median line for nonex songs
plt.axvline(non_explicit_songs.median(), color='blue', linestyle='dashed', linewidth=1, label=f'Non-Explicit Median: {non_explicit_songs.median()}')

#Histogram for explicit songs
plt.hist(explicit_songs, bins=30, alpha=0.5, label='Explicit', color='red')
#Median line for explicit songs
plt.axvline(explicit_songs.median(), color='red', linestyle='dashed', linewidth=1, label=f'Explicit Median: {explicit_songs.median()}')


plt.title('Popularity Distribution: Explicit vs. Non-Explicit Songs')
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.legend()

plt.show()

#QUESTION 4
print("Q4")
#Split data into two groups, major and minor songs
popularity = data['popularity']
mode = data['mode']  

#Separate the popularity data into two groups based on the key
popularity_major = popularity[mode == 1]
popularity_minor = popularity[mode == 0]# 1 for major key, 0 for minor key

# Perform the Mann-Whitney U Test
stat, p_value = mannwhitneyu(popularity_major, popularity_minor, alternative='greater')

# Print the results of the test
print(f'Mann-Whitney U Test: U={stat}, p-value={p_value:.10f}')

# Plot the distributions
plt.figure(figsize=(12, 6))
plt.hist(popularity_major, bins=30, alpha=0.5, label='Major Key (1)', color='blue')
plt.hist(popularity_minor, bins=30, alpha=0.5, label='Minor Key (0)', color='red')
plt.axvline(popularity_major.median(), color='blue', linestyle='dashed', linewidth=1)
plt.axvline(popularity_minor.median(), color='red', linestyle='dashed', linewidth=1)
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.title('Distribution of Popularity by Key')
plt.legend()
plt.grid(True)
plt.show()

#QUESTION 5
print("Q5")

#Plot of energy vs loudness
plt.figure(figsize=(10, 6))
sns.scatterplot(x='loudness', y='energy', data=data, alpha=0.5, color='blue')
plt.title('Effect of Loudness on Energy of Songs')
plt.xlabel('Loudness')
plt.ylabel('Energy')
plt.grid(True)
plt.show()

#Calc pearson and spearman corr coeff
pearson_corr_coeff = stats.pearsonr(data['energy'].dropna(), data['loudness'].dropna())
print("Pearson correlation coefficient:", pearson_corr_coeff)
spearman_corr_coeff = stats.spearmanr(data['energy'].dropna(), data['loudness'].dropna())
print("Spearman correlation coefficient:", spearman_corr_coeff)
print()


#QUESTION 6
print("Q6")


song_features_data = data[features] #filter dataset to only the 10 features
zscored_data = stats.zscore(song_features_data) #z-score data

#split data
x = zscored_data.values
y = data['popularity'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=N)

# Iterate through each feature and build a linear regression model
prediction = {}

for feature in features:
    x_train_feature = x_train[:, features.index(feature)].reshape(-1, 1)
    x_test_feature = x_test[:, features.index(feature)].reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_train_feature, y_train)
    y_pred = model.predict(x_test_feature)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)
    prediction[feature] = {'RMSE': rmse, 'R-squared': r_squared}
    
for feature, scores in prediction.items():
    print(f"Feature: {feature}")
    print(f"RMSE: {scores['RMSE']}")
    print(f"(COD) R^2: {scores['R-squared']}")
    print()

#Find feature with lowest RMSE and highest R^2, best predictor
low_RMSE = min(prediction, key=lambda k: prediction[k]['RMSE'])
high_R2 = max(prediction, key=lambda k: prediction[k]['R-squared'])

print(f"{low_RMSE} predicts popularity the best from the RMSE")
print(f"{high_R2} predicts popularity the best from R^2")
print()
print(f"RMSE for {low_RMSE}: {prediction[low_RMSE]}")
print(f"R^2 for {high_R2}: {prediction[high_R2]}")
print()

# instrumentalness residual plot
x_plot_feature = x_test[:, features.index(low_RMSE)].reshape(-1, 1)
plt.scatter(x_plot_feature, y_test, color='black', s=10)
plt.plot(x_plot_feature, y_pred, color='red', linewidth=0.1)
plt.xlabel(low_RMSE)
plt.ylabel('Popularity')
plt.title('Instrumentalness on Song Popularity')
plt.show()

print()

#QUESTION 7
print("Q7")

# z-score and split data
x = zscored_data.values
y = data['popularity'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=N)

#linear regression model 
model_10features = LinearRegression()
model_10features.fit(x_train, y_train)
y_pred_all_features = model_10features.predict(x_test)

#calc RMSE and R^2 
rmse_allFeatures = np.sqrt(mean_squared_error(y_test, y_pred_all_features))
r2_allFeatures = r2_score(y_test, y_pred_all_features)
print("MULTIPLE REGRESSION model for predicting popularity with all 10 features")
#Access RMSE value for best feature
best_RMSE = prediction[low_RMSE]['RMSE']
best_R2 = prediction[high_R2]['R-squared']

print(f"RMSE of model using all 10 features: {rmse_allFeatures}")
#Compare with model from question 6
RMSE_improvement = best_RMSE - rmse_allFeatures
print(f"RMSE Improvement: {RMSE_improvement}")
print(f"R^2 of  model using all 10 features: {r2_allFeatures}")
R2_improvement = r2_allFeatures - best_R2
print(f"R^2 Improvement {R2_improvement}")


print()


#QUESTION 8
print("Q8")


#STEP 1,perform the PCA and display expplained variance of each feature
#Extract and standardize the features data
X = data[features].dropna()
X_standardized = (X - X.mean()) / X.std()

# Perform PCA
pca = PCA()
pca.fit(X_standardized)

# Explained variance
explained_variance = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

#STEP 2, Display explained variance
explained_variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Explained Variance': explained_variance
})
print(explained_variance_df)

#STEP 3, CREATE SCREE PLOT WITH PCs, CUTOFF WITH KAISER (eignevalue>1)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.7, label='Eigenvalues')
plt.axhline(y=1, color='r', linestyle='-', label='Kaiser Criterion (Eigenvalue > 1)')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot with Kaiser Criterion')
plt.legend()
plt.grid()
plt.show()

# Determine the number of components with eigenvalues > 1 (Kaiser Criterion)
num_components = np.sum(eigenvalues > 1)

# Print the number of components and the proportion of variance they account for
print(f'Num of principal components with eigenvalues > 1 (Kaiser Criterion): {num_components}')
print(f'Proportion of variance explained by these components: {explained_variance[:num_components].sum():.6f}')

#STEP 4, Plot principal components, the Kaiser criteria captured 3 of them > 1
loadings = pca.components_

x = range(1, len(features) + 1)
for i_component in range(3):
    plt.bar(x, loadings[i_component, :] * -1, color='red')
    plt.xlabel('Features')
    plt.ylabel(f'loading (magnitude) on PC {i_component + 1}')
    plt.title(f'Plot for PC {i_component + 1}')
    plt.xticks(range(1, len(features) + 1), features, rotation='vertical')
    plt.grid(True)
    plt.show()
    
#QUESTION 9
print("Q9")

#PREDICTING major or minor key from valence

#extract valence and major/minor key data
#X = data[['valence']]
#CHANGED PREDICTOR FROM VALENCE TO ENERGY
X = data[['energy']]
y = data['mode']


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=N, stratify=y)

#import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#Handle class imbalance using Syntheic Minority Oversampling technique (SMOTE)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

#Standardize  data
scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_test = scaler.transform(X_test)

#BUILD LOGISTIC REGRESSION MODEL
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_sm, y_train_sm)

y_pred = log_reg.predict(X_test)
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]



print("AUROC Score:")
print(roc_auc_score(y_test, y_pred_prob))



# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_score(y_test, y_pred_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#QUESTION 10
print("Q10")


# Extract relevant features and target variable
X_duration = data[['duration']].dropna()
y_genre = data['track_genre'].dropna()

# Convert genre to binary label: 1 if classical, 0 otherwise
y = (y_genre == 'classical').astype(int)


#Split the data , use N number seed
X_train_dur, X_test_dur, y_train, y_test = train_test_split(X_duration, y, test_size=0.2, random_state=N, stratify=y)

scaler = StandardScaler()
X_train_dur = scaler.fit_transform(X_train_dur)
X_test_dur = scaler.transform(X_test_dur)

#LOGISTIC REG MODEL FOR DURATION
log_reg_dur = LogisticRegression(random_state=N)
log_reg_dur.fit(X_train_dur, y_train)

y_pred_dur = log_reg_dur.predict(X_test_dur)
y_pred_prob_dur = log_reg_dur.predict_proba(X_test_dur)[:, 1]

print("Model using Duration:")


print("ROC AUC Score:")
print(roc_auc_score(y_test, y_pred_prob_dur))



# ROC Curve
fpr_dur, tpr_dur, _ = roc_curve(y_test, y_pred_prob_dur)
plt.figure(figsize=(10, 6))
plt.plot(fpr_dur, tpr_dur, label=f'ROC curve (Duration, area = {roc_auc_score(y_test, y_pred_prob_dur):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Duration')
plt.legend(loc='lower right')
plt.grid()
plt.show()


song_features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                 'instrumentalness', 'liveness', 'valence', 'tempo']
X_features = data[song_features].dropna()
y_genre = data['track_genre'].dropna()

# Convert genre to binary label: 1 if classical is TRUE, 0 if ZFALSE
y = (y_genre == 'classical').astype(int)


X_features_standardized = scaler.fit_transform(X_features)

pca = PCA()
X_pca = pca.fit_transform(X_features_standardized)

eigenvalues = pca.explained_variance_
num_components = np.sum(eigenvalues > 1)
X_pca_selected = X_pca[:, :num_components]

#put data into training and testing sets using PCA features, use my N number
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca_selected, y, test_size=0.2, random_state=N, stratify=y)

#Build the logistic regression model using PCA components
log_reg_pca = LogisticRegression(random_state=N)
log_reg_pca.fit(X_train_pca, y_train)

# Make predictions
y_pred_pca = log_reg_pca.predict(X_test_pca)
y_pred_prob_pca = log_reg_pca.predict_proba(X_test_pca)[:, 1]

# Evaluate the model using PCA components
print("\nModel using PCA Components:")


print("ROC AUC Score:")
print(roc_auc_score(y_test, y_pred_prob_pca))

# ROC Curve
fpr_pca, tpr_pca, _ = roc_curve(y_test, y_pred_prob_pca)
plt.figure(figsize=(10, 6))
plt.plot(fpr_pca, tpr_pca, label=f'ROC curve (PCA, area = {roc_auc_score(y_test, y_pred_prob_pca):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - PCA Components')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Comparison
print(f"ROC AUC Score using Duration: {roc_auc_score(y_test, y_pred_prob_dur):.2f}")
print(f"ROC AUC Score using PCA Components: {roc_auc_score(y_test, y_pred_prob_pca):.2f}")




# EXTRACT beats per measure aka time sig and danceability
time_signature = data['time_signature']
danceability = data['danceability']


#Group by time signatures (1-5) and calc the avg danceability
avg_danceability = data.groupby('time_signature')['danceability'].mean()

avg_danceability = avg_danceability.loc[1:5]

# Plot the bar plot
plt.figure(figsize=(10, 6))
avg_danceability.plot(kind='bar', color='purple')
plt.xlabel('Time Signature (Beats per Measure)')
plt.ylabel('Average Danceability')
plt.title('Average Danceability by Time Signature')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

correlation_coefficient, p_value = pearsonr(time_signature, danceability)

print(f'Correlation Coefficient: {correlation_coefficient:.4f}')
print(f'p-value: {p_value:.10f}')
    