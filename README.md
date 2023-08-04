# DHHS Chronic Disease Indicators - Chronic Obstructive Pulmonary Disease (COPD) Prevalence Analysis
# <bu>CLASSIFICATION PROJECT</bu>
by Annie Carter
Sourced by U.S. Department of Health & Human Services
![image](https://img.freepik.com/premium-photo/lungs-airway-3d-style-darktable-processing_921860-35264.jpg?size=626&ext=jpg&ga=GA1.2.382400335.1688677589&semt=sph)

Trello: https://trello.com/invite/b/8SJiWQzt/ATTIc70085783caf23713771e813130743131BC98E7B/individual-project

## <u>Background</u>
COPD is a prevalent and chronic illness primarily caused by inhaling tobacco smoke, ranking as the third leading cause of death in the United States. COPD is a heterogeneous lung condition characterized by chronic respiratory symptoms such as dyspnea, cough, expectoration, and exacerbations, which result from abnormalities in the airways (bronchitis, bronchiolitis) and/or alveoli (emphysema), leading to persistent and often progressive airflow obstruction. According to estimates, approximately 13 million Americans have received a diagnosis of COPD, and an additional 13 million individuals are unaware of their COPD diagnosis, totaling to about 26 million Americans affected by this chronic respiratory disease (ALA, 2023).

## <u>Project Description</u>
This machine learning classification project aims to conduct an extensive analysis and predictive assessment of Chronic Obstructive Pulmonary Disease (COPD) prevalence in the United States, utilizing the comprehensive "U.S. Chronic Disease Indicators (CDI)" dataset. The dataset encompasses a wealth of COPD-related information, including risk factors, prevalence rates, health outcomes, and crucial demographic data such as, gender, race, and geographical location. A ML Classification project could shed light on the characteristics of this population and investigate health disparities among vulnerable groups. Understanding COPD's prevalence across different demographics is vital for addressing health disparities, ultimately playing a significant role in reducing the impact of COPD on public health in the United States.

## <u>Project Goal</u>

The main objective of this project is to build a predictive model using advanced ML classification techniques that leverage demographic and chronic disease indicators to address fluctuations in COPD rates, with specific attention to gender and race demographics. By conducting this analysis, we seek to identify complex patterns and crucial risk factors related to COPD prevalence. The valuable insights obtained from this investigation will inform targeted interventions, early detection approaches, and enhanced healthcare planning.


## <u>Initial Questions</u>
1. Is there a relationship between the category of "male or female" and COPD prevalence?
2. Is there a relationship between race and COPD prevalence?
3. Is there a relationship between the year and COPD prevalence?
4. Is there a relationship between the state and COPD prevalence?

## Data Dictionary

The initial dataset comprised 34 columns, which reduced to 9 columns after preparation. It contained 1,185,676 rows, but a random sample of 1,000,000 rows was chosen for this project using a random state of 42. During preparation, the central target column "Yes_COPD" was created by combining the top four prevalent chronic diseases: COPD, Cardiovascular Disease, Diabetes, and Chronic Obstructive Pulmonary Disease (COPD). Column names were renamed for improved readability and cleaned for data integrity, resulting in 537,407 rows. Some column definitions were obtained from the Center for Disease Control and Prevention (CDC) Morbidity and Mortality Weekly Report (MMWR) at https://www.cdc.gov/mmwr/pdf/rr/rr6401.pdf. 

| Original                    |   Target     |       Datatype           |       Definition              |
|-----------------------------|------------- |--------------------------|------------------------------ |
|Topic (COPD, Cardiovascular  |              |                          |                               |
|Disease, Diabetes, Chronic   |Yes_COPD      | 537407 non-null  int64   |  target variable              |
|Obstructive Pulmonary Disease|              |                          |                               |


|     Original                |   Feature     |       Datatype          |     Definition                |
|-----------------------------|-------------- |------------------------ |------------------------------ |
|YearStart                    |Year           | 537407 non-null  int64  | Year of observations          |
|LocationAbbr                 |State (Abbr)   | 537407 non-null  object | State Abbreviation            |
|Gender                       |Gender         | 537407 non-null  object | Male or Female                | 
|Stratification1              |Demographics   | 537407 non-null  object | Demographics                  |    
|GeoLocation                  |Geo Location   | 537407 non-null  object | Latitude and Longituted       |
|Race/Ethnicity               |Race/Ethnicity | 537407 non-null  object | Race or Ethnicity             |
|StratificationCategory1      |"same"         | 537407 non-null  object | Used for feature engineering  |
|Longitude                    |Longitude      | 537407 non-null  float64| Longitude                     |
|Latitude                     |Latitude       | 537407 non-null  float64| Latitude                      |
|Yes_female                   |Yes_Female     | 537407 non-null  int64  | Female = 1 Male = 0           |
|Yes_White, non-Hispanic      |Yes_White      | 537407 non-null  int64  | Yes_White =1 No= 0            |
|Yes_Black, non-Hispanic      |Yes_Black      | 537407 non-null  int64  | Yes_Black = 1, No = 0         |
|Yes_Hispanic                 |Yes_Hispanic   | 537407 non-null  int64  | Yes_Hispanic = 1 No = 0       |
|Yes_Asian or Pacific Islander|Yes_Asian_PI   | 537407 non-null  int64  | Yes_Asian_PI = 1, No = 0      |
|Yes_Other, non-Hispanic      |Yes_Other      | 537407 non-null  int64  | Yes_Other = 1, No =0          |
|Yes_Amn Indian Alaska Native |Yes_Native_Amn | 537407 non-null  int64  | Yes_Native = 1, No =0         |
|Yes_Multiracial, non-Hispanic|Yes_Multiracial| 537407 non-null  int64  | Yes_Multiracial = 1, No =0    |


## <u>Statistical Testing Hypothesis </u>
Hypothesis 1 - 

alpha = .05
* H0 =  Category of "male or female" gender has no relationship to COPD
* Ha = Category of "male or female" gender has a relationship to COPD
* Outcome: We reject the Null Hypothesis.

Hypothesis 2 - 

alpha = .05
* H0 = Race has no relationship to COPD  prevalence
* Ha = Race has a relationship to COPD  prevalence
* Outcome: We reject the Null Hypothesis.

Hypothesis 3 -

alpha = .05
* H0 = Year has no relationship to COPD prevalence
* Ha = Year has a relationship to COPD prevalence
* Outcome: We reject the Null Hypothesis.

## <u>Planning Process</u>
#### Planning
1. Clearly define the problem statement related to the DHHS Chronic Disease Indicator dataset, including essential information for addressing the chronic disease of interest.

2. Create a detailed README.md file documenting the project's context, dataset characteristics, and analysis procedure for easy reproducibility.

#### Acquisition and Preparation
3. Preprocess the data, handling missing values and outliers effectively during data loading and cleansing.

4. Perform feature selection meticulously, identifying influential features impacting the prevalence of the chronic disease through correlation analysis, feature importance estimation, or domain expertise-based selection criteria.

5. Develop specialized scripts (e.g., acquire.py and wrangle.py) for efficient and consistent data acquisition, preparation, and data splitting.

6. Safeguard proprietary aspects of the project by implementing confidentiality and data security measures, using .gitignore to exclude sensitive information.

#### Exploratory Analysis
7. Utilize exploratory data analysis techniques, employing compelling visualizations and relevant statistical tests to extract meaningful patterns and relationships within the dataset.

#### Modeling
8. Carefully choose a suitable machine learning algorithm, evaluating options like Logistic Regression, Decision Trees, Random Forests, or K Nearest Neighbor tailored for the regression task.

9. Implement the selected machine learning models using robust libraries (e.g., scikit-learn), systematically evaluating multiple models, including Decision Trees, Logistic Regression, and Random Forests, with a fixed Random Seed value 42 for reproducibility.

10. Train the models rigorously to ensure optimal learning and model performance.

11. Conduct rigorous model validation techniques to assess model generalization capability and reliability.

12. Select the most effective model, such as Logistic Regression, based on thorough evaluation metrics for further analysis.

#### Product Delivery
13. Assemble a final notebook, combining superior visualizations, well-trained models, and pertinent data to present comprehensive insights and conclusions with scientific rigor.

14. Generate a Prediction.csv file containing predictions from the chosen model on test data for further evaluation and utilization.

15. Maintain meticulous project documentation, adhering to scientific and professional standards, to ensure successful presentation or seamless deployment.

## <u>Instructions to Reproduce the Final Project Notebook</u> 
To successfully run/reproduce the final project notebook, please follow these steps:

1.  Read this README.md document to familiarize yourself with the project details and key findings.
2. Before proceeding, ensure that you have the necessary database credentials. Get data set from https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi Create .gitignore for privacy if necessary
3. Clone the classification_project repository from my GitHub or download the following files: aquire.py, wrange.py or prepare.py, and final_report.ipynb. You can find these files in the project repository.
4. Open the final_report.ipynb notebook in your preferred Jupyter Notebook environment or any compatible Python environment.
5. Ensure that all necessary libraries or dependent programs are installed. You may need to install additional packages if they are not already present in your environment.
6. Run the final_report.ipynb notebook to execute the project code and generate the results.
By following these instructions, you will be able to reproduce the analysis and review the project's final report. Feel free to explore the code, visualizations, and conclusions presented in the notebook.


## <u>Key Findings</u>

* <span style ='color:#1F456E'> Gender Disparity in COPD Diagnosis: Women had higher observations of COPD than men. This could be because women smokers are about 50% more likely to develop COPD than men. Generally, women smoke less than men, suggesting that they may be more susceptible to developing COPD. In another large population study, females appear to have more severe COPD with early-onset disease and a greater susceptibility to COPD with lower tobacco exposure. (Barnes,2016)
    
* <span style ='color:#1F456E'> This analysis revealed significant gender disparities in COPD observations of female to male. This may be attributed to men having higher rates of undiagnosed COPD, irrespective of race. This suggests that men are less likely to receive timely COPD diagnoses compared to women (Mamary et al., 2018). Resulting in females having higher reported cases of COPD. An outside study suggests, one possible factor could be differences in symptom perception and attitudes toward medical care between genders(Mamary et al., 2018). Which also could show why women have higher diagnosis. Addressing this gender disparity in COPD diagnosis is crucial to ensure timely interventions and improved healthcare outcomes for both men and women.
    
* <span style ='color:#1F456E'> The data highlights racial disparities in COPD diagnosis. When considering the population distribution by in the US by race 60% White, 18 % Hispanic, 12% African American (AA) and 6% Asian. COPD is unevenly distributed with AA, and Hispanic having the same amount as White dispite having a smaller populous. Once considered a disease primarily affecting white men, now recognized as increasingly prevalent among AA men and women. One study showed that the risk for undiagnosed COPD was not uniform within the study population, indicating significant disparities by race . Understanding the population characteristics of the approximately 13 million individuals with undiagnosed COPD is essential to address and mitigate these disparities(Mamary et al., 2018). Tailored interventions and targeted healthcare initiatives should be implemented to improve COPD diagnosis rates and healthcare access for racial minority groups. (Mamary et al., 2018)

* <span style ='color:#1F456E'> Yearly Rates in COPD: Evidence showed substantial shift in COPD prevalence From 2014 to 2018, dispite COPD rates remaining stable, with little change on average year to year. However, a significant decline in both rates and counts occurred from 2018 to 2019, possibly due to a change in the question format and later the COVID-19 global pandemic. The subsequent decrease in COPD rates and counts from 2019 to 2020 may not be statistically relevant.(ALA, 2023; Awatade, 2023)
The similarity in symptoms between COPD and COVID-19 has resulted in delayed diagnosis for some COPD patients infected with COVID-19. Misdiagnosis as a COPD exacerbation has been reported, further complicating the reporting of COPD and may be reason for reported decline. (Awatade, 2023).
    
    
## <u>Conclusion</u>
- The analysis of the DHHS Chronic Disease Indicators dataset revealed significant relationships between gender, race/ethnicity, US locations, and COPD prevalence.
- The predictive models, including the Decision Tree Test Model, consistently performed well with close alignment to the baseline accuracy of 76.21%. The Decision Tree Test Model showed a marginal improvement at 76.28%.
- The findings underscore gender and racial disparities in COPD diagnosis, highlighting the need for tailored interventions to improve healthcare access and timely diagnoses for underserved populations. Continuous monitoring of COPD rates is crucial for targeted interventions in COPD prevention and management. Additionally, the impact of COVID-19 on COPD patients emphasizes the importance of considering comorbidities and ensuring prompt and accurate diagnoses during pandemics. This study provides essential insights into COPD prevalence, guiding effective public health strategies to mitigate COPD's impact in the United States.

## <u>Next Steps</u>

1. **Time-Series Analysis:** Consider conducting a time-series analysis to gain valuable insights into the trends and patterns of COPD prevalence over the years. Pay particular attention to the significant decline observed between 2018 and 2019. Exploring this time range in more detail may reveal underlying factors or interventions that contributed to the change in COPD rates.

2. **Melt Observation Data:** Refine the observations and uncover more meaningful relationships by utilizing the 'melt' operation to reshape the data. This transformation converts the data from a wide format to a long format, facilitating the analysis and visualization of relationships between different variables and COPD prevalence.

3. **Geo-Location Clustering:** Investigate the spatial distribution of COPD prevalence by selecting specific geographic areas. Conduct clustering analysis to identify regions with similar COPD patterns based on geo-location data. This exploration can shed light on whether certain locations are more susceptible to higher or lower COPD rates and guide targeted intervention strategies.

4. **Feature Engineering for Specific COPD Types and Age:** Enhance predictions and insights by implementing targeted feature engineering for specific COPD types and age groups in the DHHS Chronic Disease Indicators analysis. Capture unique characteristics and risk factors associated with different COPD types and specific age ranges to improve accuracy in prevalence predictions.
    
## <u>Recommendations</u>

- **Targeted Awareness Campaigns:** 
-- Implement targeted interventions to address gender disparities in COPD diagnosis, encouraging early reporting of symptoms and enhancing healthcare access for both men and women.
--Develop and implement tailored healthcare initiatives to address racial disparities in COPD diagnosis, focusing on improving access and healthcare outcomes for racial minority groups.
-- Enhance awareness and education among healthcare providers to differentiate COVID-19 symptoms from COPD exacerbation to enable prompt and accurate diagnosis for COPD patients during pandemics.Focus on raising awareness about specific COPD types that are most prevalent in certain gender and race/ethnicity groups. Tailored awareness campaigns can improve early detection and prompt appropriate interventions.

- **Further Research:** Conduct further research to understand the reasons there has been no improvemetnt with COPD. Possible decrease in cigarettes, but increase in vaping. Identify factors contributing to stagnation in COPD prevalence.

By implementing these recommendations and conducting additional research, we can gain deeper insights into COPD prevalence, improve early detection, and implement effective interventions, ultimately leading to better COPD outcomes and improved public health.
    
    
## <u>LinkedIn Project Description</u>
    
A comprehensive investigation was conducted on COPD, utilizing machine learning techniques with the DHHS Chronic Disease Indicator dataset, which contained over 1.2 million entries. To process and visualize the data effectively, I employed essential Python libraries such as Pandas, Matplotlib, Seaborn, Scipy, and Scikit-Learn. For presenting the geographical distribution, I used Tableau to create interactive map visualizations. Additionally, I incorporated Folium to produce dynamic and engaging map visuals for an interactive user experience. During the analysis, I opted for the Decision Tree model due to its consistent and marginally beat baseline accuracy significant findings, which were further supported by peer-reviewed studies on COPD. The primary goal of the project was to contribute to public health by enhancing the understanding of COPD prevalence management and identifying potential improvements.
    
    
## <u>References</u>

- American Lung Association.[ALA] (2023). COPD Prevalence. Retrieved from https://www.lung.org/research/trends-in-lung-disease/copd-trends-brief/copd-prevalence
- Awatade, N. T., Wark, P. A. B., Chan, A. S. L., Mamun, S. M. A. A., Mohd Esa, N. Y., Matsunaga, K., Rhee, C. K., Hansbro, P. M., Sohal, S. S., & On Behalf Of The Asian Pacific Society Of Respirology Apsr Copd Assembly (2023). The Complex Association between COPD and COVID-19. Journal of clinical medicine, 12(11), 3791. https://doi.org/10.3390/jcm12113791
- Barnes, P. J. (2016). Sex differences in chronic obstructive pulmonary disease mechanisms. American journal of respiratory and critical care medicine, 193(8), 813-814.
- Mamary, A. J., Stewart, J. I., Kinney, G. L., Hokanson, J. E., Shenoy, K., Dransfield, M. T., Foreman, M. G., Vance, G. B., Criner, G. J., & COPDGene® Investigators (2018). Race and Gender Disparities are Evident in COPD Underdiagnoses Across all Severities of Measured Airflow Obstruction. Chronic obstructive pulmonary diseases (Miami, Fla.), 5(3), 177–184. https://doi.org/10.15326/jcopdf.5.3.2017.0145
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6296789/
    
