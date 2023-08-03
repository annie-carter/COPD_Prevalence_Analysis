#IMPORT LIBRARIES
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import os
import folium
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr


# import Machine Learning Library for classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import datetime

import warnings
warnings.filterwarnings("ignore")


#---------- ACQUIRE & PREPARE-------
def prep_copd():
    ''' 
     The below functions prepares DHSS CDI for COPD prevalance analysis 
    '''
    # Save and read dataset csv from https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi
    df = pd.read_csv('U.S._Chronic_Disease_Indicators__CDI_.csv')
    
    #created sample DF with random state of 42 to review and clean data rapidly
    df_sample= df.sample(n=1000000, random_state=42)
    
    # List of columns to remove from Dataframe. 
    columns_to_remove = ['YearEnd', 'Response', 'StratificationCategory2', 'Stratification2', 'StratificationCategory3', 'DataValue',
                     'Stratification3', 'ResponseID', 'StratificationCategoryID2', 'StratificationID2',
                     'StratificationCategoryID3', 'StratificationID3','DataValueTypeID','QuestionID', 'TopicID','LocationID','HighConfidenceLimit','LowConfidenceLimit','YearEnd','LocationDesc','DataValueUnit','DataValueType','DataValueAlt','DataValueFootnoteSymbol','DatavalueFootnote','StratificationCategoryID1','StratificationID1','Question','DataSource']
    # Drop unnecessary columns from the Dataframe
    df_sample = df_sample.drop(columns_to_remove, axis=1)
    
     #change column names to be more readable
    df_sample = df_sample.rename(columns={'YearStart':'Year', 'Stratification1':'Demographics','GeoLocation':'Geo Location', 'LocationAbbr' : 'State Abbr','Topic': 'Disease'})
    
    # List of values to remove from the 'Topic' column
    values_to_remove = ['Asthma', 'Arthritis', 'Nutrition, Physical Activity, and Weight Status', 'Overarching Conditions','Alcohol','Tobacco','Chronic Kidney Disease','Older Adults','Oral Health','Mental Health','Immunization','Reproductive Health','Disability']

    # Extract latitude and longitude from 'Geo Location' column
    df_sample[['Longitude', 'Latitude']] = df_sample['Geo Location'].str.extract(r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)')
    # Convert the latitude and longitude values to float
    df_sample['Longitude'] = df_sample['Longitude'].astype(float)
    df_sample['Latitude'] = df_sample['Latitude'].astype(float)
    
    # Drop rows with specific values from the 'Topic' column
    df_sample = df_sample.drop(df_sample[df_sample['Disease'].isin(values_to_remove)].index)
    
    

    ''' Will use COPD to create one-hot code "dummy" value for prevalaence "Yes_COPD" and Cardiovascular Disease, Diabetes & COPD. I will remove other Topics column '''
    # Create a dummy variable for the 'Yes_COPD' column
    df_sample['Yes_COPD'] = np.where(df_sample['Disease'] == 'Chronic Obstructive Pulmonary Disease', 1, 0).astype(int)
    # Drop the original 'Disease' column
    df_sample.drop('Disease', axis=1, inplace=True)
    
    # Create a new column 'Race/Ethnicity' based on the condition
    df_sample['Race/Ethnicity'] = np.where(df_sample.StratificationCategory1 == 'Race/Ethnicity', df_sample.Demographics, '')

    # Create a new column 'Race/Gender' based on the condition
    df_sample['Gender'] = np.where(df_sample.StratificationCategory1 == 'Gender', df_sample.Demographics, '')
    
    # Will use Female to create one-hot code "dummy" value for "female" 
    df_sample['Yes_female'] = np.where(df_sample['Gender'] == 'Female', 1, 0).astype(int)

    #Remove nulls
    df_sample.dropna(inplace=True)
    
    ''' This function creates a csv '''
    cdi = df_sample

    # Save the DataFrame to a CSV file
    df_sample.to_csv("COPD.csv", index=False)  

    filename = 'COPD.csv'
    if os.path.isfile(filename):
        pd.read_csv(filename)
    return df_sample

def demographic_graph(df_sample):
    # Get the value counts of 'COPD' topic in the 'Demographics' column
    demo_copd = df_sample[df_sample['Yes_COPD'] == 1]['Demographics'].value_counts()
    
    # Create a bar plot using Seaborn
    plt.figure(figsize=(12, 10))
    dc = sns.barplot(x=demo_copd.index, y=demo_copd.values, palette='Blues')
    
    # Set labels and title
    plt.xlabel('Demographics')
    plt.ylabel('Count')
    plt.title('Value Counts of "COPD" based on Demographics')
              
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    
    # Add count numbers on bars
    for p in dc.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()    
        offset = width * 0.02  # Adjust the offset percentage as needed
        dc.annotate(format(height, '.0f'), (x + width / 2., y + height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    # Show the plot
    plt.tight_layout()  
    plt.show()
#-------------- SPLIT-------
def split_sample(df_sample):
    ''' The below functions were created in regression excercises and will be aggregated to make a master clean_data function for final 
        report
    '''
    train_validate, sample_test = train_test_split(df_sample, test_size=0.2, random_state=42)
    sample_train, sample_validate = train_test_split(train_validate, test_size=0.25, random_state=42)
    print(f'Train shape: {sample_train.shape}')
    print(f'Validate shape: {sample_validate.shape}')
    print(f'Test shape: {sample_test.shape}')
    return sample_train, sample_validate, sample_test 

def X_y_split(sample_train, sample_validate, sample_test):
    #Splitting the data in to X and Y to take out the data with curn and those without 
    sample_X_train = sample_train.select_dtypes(exclude=['object']).drop(columns=['Yes_COPD'])
    sample_y_train = sample_train.select_dtypes(exclude=['object']).Yes_COPD
    
    sample_X_validate = sample_validate.select_dtypes(exclude=['object']).drop(columns=['Yes_COPD'])
    sample_y_validate = sample_validate.select_dtypes(exclude=['object']).Yes_COPD
    
    sample_X_test = sample_test.select_dtypes(exclude=['object']).drop(columns=['Yes_COPD'])
    sample_y_test = sample_test.select_dtypes(exclude=['object']).Yes_COPD
    #------------GENDER VS COPD--------

def gender_graph(sample_train):
    ''' This functions graphs gender vs COPD''' 
    # Create DataFrame for graph
    gender_graph = pd.DataFrame(sample_train)
    
    # Create DataFrame for graph
    gender_graph_df = sample_train[sample_train['Gender'].isin(['Male', 'Female'])].dropna(subset=['Gender'])
    
    # Assuming you have a DataFrame 'sample_train' with the required data
    new_labels = {'no COPD': 'No COPD', 'COPD': 'COPD'}
    
    # Set a larger figure size
    plt.figure(figsize=(10, 6))
    
    # Visualizing the Gender vs COPD
    gg = sns.countplot(data=gender_graph_df, x='Gender', hue='Yes_COPD', palette='Blues')
    
    # Access the legend object
    legend = gg.legend()
    
    # Modify the legend labels
    legend.get_texts()[0].set_text(new_labels['no COPD'])
    legend.get_texts()[1].set_text(new_labels['COPD'])
    
    gg.set_xlabel('Gender')
    gg.set_ylabel('Number of Observations')
    plt.title('How Gender Relates to COPD?')
    
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    
    # Add count numbers on bars
    for p in gg.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()    
        offset = width * 0.02  # Adjust the offset percentage as needed
        gg.annotate(format(height, '.0f'), (x + width / 2., y + height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    # Use tight layout
    plt.tight_layout()   
    plt.show()

def gender_observed(sample_train):
    ''' This functions graphs Observed vs COPD''' 
    alpha = 0.05    
    gender_observed = pd.crosstab(sample_train.Yes_COPD, sample_train.Yes_female)
    # Assuming you have a DataFrame 'sample_train' with the required data
    new_labels = {'no COPD': 'No COPD', 'COPD': 'COPD'}
    # Plot the observed data as a bar plot
    go =gender_observed.plot(kind='bar', stacked=True, color=['pink','skyblue' ], edgecolor='black')
    # Access the legend object
    legend = go.legend()
        
    # Modify the legend labels
    legend.get_texts()[0].set_text(new_labels['no COPD'])
    legend.get_texts()[1].set_text(new_labels['COPD'])
    
    # Set the labels and title
    plt.xlabel('COPD Status')
    plt.ylabel('Count')
    plt.title('Observed COPD Status by Gender')
    plt.legend(title='Gender', loc='upper right', labels=['Female', 'Male'])
    
    # Rename the x-axis labels
    go.set_xticklabels(['No COPD', 'Yes COPD'], rotation=0)
    
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=0)
        
    # Add count numbers on bars
    for p in go.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()    
        offset = width * 0.02  # Adjust the offset percentage as needed
        go.annotate(format(height, '.0f'), (x + width / 2., y + height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
    # Use tight layout
    plt.tight_layout()
    plt.show()
    
def gender_stat(sample_train):
    ''' This functions chi-square stat for gender''' 
    alpha = 0.05
    gender_observed = pd.crosstab(sample_train.Yes_COPD, sample_train.Yes_female)
    stats.chi2_contingency(gender_observed)
    chi2, p, degf, expected = stats.chi2_contingency(gender_observed)
    print('Gender Observed')
    print(gender_observed.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p_value = {p:.4f}')
    print(f'The p-value is less than the alpha: {p < alpha}')
    if p < alpha:
        print('We reject the null')
    else:
        print("we fail to reject the null")

#------- RACE VS COPD-----

def race_graph(sample_train):
    ''' This function creates a dataframe for Race/Ethnicity and then uses it to graphs'''
    # Create DataFrame for graph
    race_graph_df = pd.DataFrame(sample_train)
    
    # Filter the DataFrame to keep only 'Male' and 'Female' values and drop rows with blank values
    race_graph_df = race_graph_df[race_graph_df['Race/Ethnicity'].isin(['White, non-Hispanic','Black, non-Hispanic', 'Hispanic', 'Asian or Pacific Islander', 'American Indian or Alaska Native', 'Other, non-Hispanic','Multiracial, non-Hispanic'])].dropna(subset=['Race/Ethnicity'])
    
    #relabel
    new_labels = {'no COPD': 'No COPD', 'COPD': 'COPD'}
    
    # Set a larger figure size
    plt.figure(figsize=(10, 6))
    
    # Visualizing the Race/Ethnicity vs COPD
    eg = sns.countplot(data=race_graph_df, x='Race/Ethnicity', hue='Yes_COPD', palette='Blues')
    
    # Access the legend object
    legend = eg.legend()
    
    # Modify the legend labels
    legend.get_texts()[0].set_text(new_labels['no COPD'])
    legend.get_texts()[1].set_text(new_labels['COPD'])
    
    eg.set_xlabel('Race/Ethnicity')
    eg.set_ylabel('Number of Observations')
    plt.title('Race/Ethnicity vs COPD')
    
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    
    # Add count numbers on bars
    for p in eg.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()    
        offset = width * 0.02  # Adjust the offset percentage as needed
        eg.annotate(format(height, '.0f'), (x + width / 2., y + height), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    # Use tight layout
    plt.tight_layout() 
    plt.show()
    
    
def race_observed(sample_train):
    ''' This function creates 3 plots (Bar, Stacked Bar and Heatmap) for Race/Ethnicity '''
    race_observed = pd.crosstab(sample_train['Yes_COPD'], sample_train['Race/Ethnicity'])
    
    # Filter the DataFrame to keep only the specified race/ethnicity values and drop rows with missing values
    race_observed_df = race_observed.dropna(subset=['White, non-Hispanic', 'Black, non-Hispanic', 'Hispanic', 'Asian or Pacific Islander', 'American Indian or Alaska Native', 'Other, non-Hispanic', 'Multiracial, non-Hispanic'])
    
    # --- Classification Plot 1: Bar Plot ---
    plt.figure(figsize=(8, 6))
    race_observed_df.plot(kind='bar', edgecolor='black')
    plt.xlabel('COPD Status')
    plt.ylabel('Count')
    plt.title('COPD Status by Race/Ethnicity')
    plt.legend(title='Race/Ethnicity', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # --- Classification Plot 2: Stacked Bar Plot ---
    plt.figure(figsize=(8, 6))
    race_observed_df.plot(kind='bar', stacked=True, edgecolor='black')
    plt.xlabel('COPD Status')
    plt.ylabel('Count')
    plt.title('COPD Status by Race/Ethnicity')
    plt.legend(title='Race/Ethnicity', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # --- Classification Plot 3: Heatmap ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(race_observed_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Race/Ethnicity')
    plt.ylabel('COPD Status')
    plt.title('COPD Status by Race/Ethnicity (Heatmap)')
    plt.tight_layout()
    plt.show()
    

def race_stats(sample_train):
    ''' This functions chi-square stat for race''' 
    alpha = 0.05
    race_observed = pd.crosstab(sample_train['Yes_COPD'], sample_train['Race/Ethnicity'])
    stats.chi2_contingency(race_observed)
    chi2, p, degf, expected = stats.chi2_contingency(race_observed)
    print('Race Observed')
    print(race_observed.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p-value = {p:.4f}')
    print(f'The p-value is less than the alpha: {p < alpha}')
    if p < alpha:
        print('We reject the null')
    else:
        print("we fail to reject the null")
        
#------- YEAR VS COPD-----        
def year_graph(sample_train):
    no_COPD_df = sample_train[sample_train['Yes_COPD'] == 0]

    # Group by 'Year' and count the number of 'No_COPD' occurrences for each year
    no_COPD_totals_by_year = no_COPD_df.groupby('Year').size()
    
    # Filter the DataFrame for rows where 'Yes_COPD' is equal to "1" (Yes COPD)
    yes_COPD_df = sample_train[sample_train['Yes_COPD'] == 1]
    
    # Group by 'Year' and count the number of 'Yes_COPD' occurrences for each year
    yes_COPD_totals_by_year = yes_COPD_df.groupby('Year').size()
    
    # Create a time-line graph for the COPD totals over the years
    plt.figure(figsize=(10, 6))
    plt.plot(no_COPD_totals_by_year.index, no_COPD_totals_by_year.values, marker='o', linestyle='-', color='g', label='No COPD')
    plt.plot(yes_COPD_totals_by_year.index, yes_COPD_totals_by_year.values, marker='o', linestyle='-', color='b', label='Yes COPD')
    plt.title('Relationship of Year with COPD')
    plt.xlabel('Year')
    plt.ylabel('COPD Totals')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
def year_stat(sample_train, sample_validate):
    '''Pearson R stat for year'''
    alpha = 0.05
    train_r, train_p = pearsonr(sample_train.Year, sample_train.Yes_COPD)
    validate_r, validate_p = pearsonr(sample_validate.Year, sample_validate.Yes_COPD)
    #    test_r, test_p = pearsonr(test.chlorides, test.quality)
    print('train_r:', train_r)
    print('train_p:',train_p)
    print('validate_r:', validate_r)
    print('validate_p:', validate_p)
    print(f'The p-value is less than the alpha: {validate_p < alpha}')
    if validate_p < alpha:
        print('Outcome: We reject the null')
    else:
        print("Outcome: We fail to reject the null")

#------- US GEO LOCATION  VS COPD-----
def map_graph(sample_train):
    map_sample = sample_train.sample(10000)
    map_sample.drop(map_sample[map_sample['Geo Location'] == ''].index, inplace=True)
    
    map_sample.dropna(inplace=True)
    # Create a folium map centered at the USA
    map_usa = folium.Map(location=[37.0902, -95.7129], zoom_start=4)
    
    # Get the count of 'Yes' for each state
    yes_count_per_state = map_sample[map_sample['Yes_COPD'] == 1].groupby('State Abbr').size()
    
    # Add markers to the map for each state with COPD values
    for idx, row in map_sample.iterrows():
        # Convert 1 to 'Yes' and 0 to 'No'
        COPD_status = 'Yes' if row['Yes_COPD'] == 1 else 'No'
        
        # Get the count of 'Yes' for the current state
        count_for_state = yes_count_per_state.get(row['State Abbr'], 0)
        
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['State Abbr']} State # of {COPD_status} COPD Observations:  {count_for_state}",
            tooltip=row['State Abbr'],
            icon=folium.Icon(icon='info-sign')
        ).add_to(map_usa)
    #saves map HTML image 
    map_usa.save("map_usa.html")
    # Display the map
    map_usa
        
    


#---------TRAIN MODELS-------------------

def train_models(sample_X_train, sample_y_train, sample_X_validate, sample_y_validate, sample_X_test, sample_y_test):
    #### Decision Tree Train
    #Make 
    sample_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    #Fit
    sample_tree = sample_tree.fit(sample_X_train, sample_y_train)
    plt.figure(figsize=(13, 7))
    plot_tree(sample_tree, feature_names=sample_X_train.columns, rounded=True)
    #Dataframe of predictions
    COPD_y_prediction = pd.DataFrame({'COPD': sample_y_train,'Baseline': 0, 'Model_1':sample_tree.predict(sample_X_train)})
    y_prediction_prob = sample_tree.predict_proba(sample_X_train)
    print(y_prediction_prob[0:5])
    print('Accuracy of Decision Tree classifier on training set: {:.4f}'
      .format(sample_tree.score(sample_X_train, sample_y_train)))
    confusion_matrix(COPD_y_prediction.COPD, COPD_y_prediction.Model_1)
    print(classification_report(COPD_y_prediction.COPD,COPD_y_prediction.Model_1))

# def log_train(sample_X_train,sample_y_train):
    ##### Logistic Regression Train
    # Make
    log_sample = LogisticRegression(C=1, random_state=42)
    #Fit 
    log_sample.fit(sample_X_train, sample_y_train)
    # Use
    y_prediction = log_sample.predict(sample_X_train)
    COPD_y_prediction['Model_2'] = y_prediction
    print('Accuracy of Logistic Regression training set: {:.4f}'
      .format(log_sample.score(sample_X_train, sample_y_train)))
    confusion_matrix(COPD_y_prediction.COPD, COPD_y_prediction.Model_2)
    print(classification_report(COPD_y_prediction.COPD,COPD_y_prediction.Model_2))
    
# def random_train(sample_X_train,sample_y_train):
    ### Random Forest Train
    #Make 
    rf_sample = RandomForestClassifier(bootstrap=True, 
                                class_weight=None, 
                                criterion='gini',
                                min_samples_leaf=1,
                                n_estimators=100,
                                max_depth=10, 
                                random_state=42)
    #Fit
    rf_sample.fit(sample_X_train, sample_y_train)
    #Use
    rf_sample.score(sample_X_train,sample_y_train)
    rf_y_prediction = rf_sample.predict(sample_X_train)
    COPD_y_prediction['Model_3'] = rf_y_prediction
    print('Accuracy of Random Forest training set: {:.4f}'
      .format(rf_sample.score(sample_X_train, sample_y_train)))
    confusion_matrix(COPD_y_prediction.COPD, COPD_y_prediction.Model_3)
    print(classification_report(COPD_y_prediction.COPD,COPD_y_prediction.Model_3))
    COPD_y_prediction.head()
    
    
#------------VALIDATE MODELS------------
# def validate_models(sample_X_validate, sample_y_validate):
    #### Decision Tree Validate
    #Dataframe of validate predictions
    COPD_y_val_prediction = pd.DataFrame({'COPD': sample_y_validate,'Baseline': 0, 'Model_1':sample_tree.predict(sample_X_validate)})
    COPD_y_val_prediction_prob = sample_tree.predict_proba(sample_X_validate)
    print(COPD_y_val_prediction_prob[0:5])
    print('Accuracy of Decision Tree validation set: {:.4f}'
          .format(sample_tree.score(sample_X_validate, sample_y_validate)))
    confusion_matrix(COPD_y_val_prediction.COPD, COPD_y_val_prediction.Model_1)
    print(classification_report(COPD_y_val_prediction.COPD,COPD_y_val_prediction.Model_1))
    
    ##### Logistic Regression Validate
    # USE
    COPD_val_y_prediction = log_sample.predict(sample_X_validate)
    COPD_y_val_prediction['Model_2'] = COPD_val_y_prediction
    print('Accuracy of Logistic Regression validation set: {:.4f}'
          .format(log_sample.score(sample_X_validate, sample_y_validate)))
    confusion_matrix(COPD_y_val_prediction.COPD, COPD_y_val_prediction.Model_2)
    print(classification_report(COPD_y_val_prediction.COPD, COPD_y_val_prediction.Model_2))
    
    #### Random Forest Validate
    #score on my train data
    rf_sample.score(sample_X_validate,sample_y_validate)
    # use the model to make predictions
    COPD_val_rf_y_prediction = rf_sample.predict(sample_X_validate)
    COPD_y_val_prediction['Model_3'] =COPD_val_rf_y_prediction
    print('Accuracy of Random Forest validation set: {:.4f}'
          .format(rf_sample.score(sample_X_validate, sample_y_validate)))
    confusion_matrix(COPD_y_val_prediction.COPD, COPD_y_val_prediction.Model_3)
    print(classification_report(COPD_y_val_prediction.COPD, COPD_y_val_prediction.Model_3))
 #---------- TEST MODEL-----------------
# def test_model(ample_X_test, sample_y_test):
    #Dataframe of validate predictions
    COPD_test_prediction = pd.DataFrame({'COPD': sample_y_test,'Baseline': 0, 'Model_1':sample_tree.predict(sample_X_test)})
    COPD_test_prediction_prob = sample_tree.predict_proba(sample_X_test)
    print(COPD_test_prediction_prob[0:5])
    print('Accuracy of Decision Tree classifier on Test set: {:.4f}'
      .format(sample_tree.score(sample_X_test, sample_y_test)))
    confusion_matrix(COPD_test_prediction.COPD, COPD_test_prediction.Model_1)
    print(classification_report(COPD_test_prediction.COPD,COPD_test_prediction.Model_1))
    