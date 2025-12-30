# Movie-Rating-Prediction-Analysis
This is a rework of the Capstone Project for the DS112 (Intro to Data Science) course. In this project, I'm analyzing the relationships between movie ratings and viewers' preferences and oersonality traits, and finally building prediction models for movie ratings based on viewers' characteristics. 

## Project Overview
From the project description provided by instructor:  "The cover story is that you are working for a major entertainment corporation that is trying to get a better handle on both what makes a good movie as well as a better understanding of the viewers." 

## Dataset Description
This dataset features ratings data of 400 movies from 1097 research participants and is contained in the file “movieReplicationSet.csv”. It is organized as follows:
1st row: Headers (Movie titles/questions) – (note that the indexing in this list is from 1)\
**Row 2-1098**: Responses from individual participants\
**Columns 1-400**: These columns contain the ratings for the 400 movies (0 to 4, and missing)\
**Columns 401-420**: These columns contain self-assessments on sensation seeking behaviors (1-5)\
**Columns 421-464**: These columns contain responses to personality questions (1-5)\
**Columns 465-474**: These columns contain self-reported movie experience ratings (1-5)\
**Column 475**: Gender identity (1 = female, 2 = male, 3 = self-described)\
**Column 476**: Only child (1 = yes, 0 = no, -1 = no response)\
**Column 477**: Social viewing preference – “movies are best enjoyed alone” (1 = y, 0 = n, -1 = nr)\


## Content Overview
- [EDA and Data Cleaning](#eda-and-data-cleaning)
- [Dimension Reduction](#dimension-reduction)
- Answering the Questions
  - [Sensation Seeking and Movie Experience](#sensation-seeking-and-movie-experience)
  - [Personality Types](#personality-types)
  - [Movie Ratings and Viewer Characteristics](#movie-ratings-and-viewer-characteristics)
  - [Franchise Movie Ratings Consistency](#franchise-movie-ratings-consistency)
  - [Prediction Model: Decision Tree Regressor with 5-fold cross-validation and GridSearchCV](#prediction-model:-decision-tree-regressor-with-5--fold-cross--validation-and-GridSearchCV)

## EDA and Data Cleaning
The dataset could be generally divided into 3 main parts: movie ratings, viewer characteristics, viewer demographics.\
We start by investigating the dataset (EDA), which indicated that a lot of the rows are consisted of missing values (total of 327526
NaNs). 

For missing values for movie ratings (columns 0-399), we will impute them with the row average, which will be the respondent's average movie ratings. One row [896] had all nulls, so we simply remove this row. For other movie ratings, they are rounded into the nearest 0.5 values.

For behavioral questions (400-473), we also use rounded mean value for the respondent.  

## Dimension Reduction
For columns 401 - 474, we have 73 variables for the viewer's characteristics and their viewership preference. To understand these variables better, I used the following steps to reduce the dimension: 
1. EDA for each type of variables: Sensation Seeking, Personality, and Movie Experience
2. Impute missing values with median value (3)
3. Examine the raw data through correlation heatmap
4. Apply PCA to transform the data into its Principal Components (PCs), reducing dimensionality while retaining most variance

### Correlation Heatmaps
<img width="300" height="200" alt="sensation_seeking_corrheatmap" src="https://github.com/user-attachments/assets/4e2f04c4-48ea-48f0-815c-9cdc938baf79" /> <img width="300" height="200" alt="personality_corrheatmap" src="https://github.com/user-attachments/assets/4120eca2-185e-4ed1-bb3c-6672c030bf7b" /> <img width="300" height="200" alt="movexp_corrheatmap" src="https://github.com/user-attachments/assets/44acdd36-7318-4859-9664-1c984af1b50b" />
From the heatmap, we can see a few variables that are highly correlated with each other, indicating PCA is appropriate. 

### Scree Plot and Transformation
<img width="300" height="200" alt="Scree1" src="https://github.com/user-attachments/assets/5b158b5b-85d9-4366-ad77-5e3b198db56f" /> <img width="300" height="200" alt="scree3" src="https://github.com/user-attachments/assets/66e17326-ec16-4a07-9990-d16b7f2f572e" /><img width="300" height="200" alt="Scree2" src="https://github.com/user-attachments/assets/3c526915-d57e-4616-a0a5-d2867e52b17a" />
Using Scree Plot, we look at the eigenvalues (variance explained) for each component, and decide how many components to keep based on Kaiser and elbow criterion. 

To further investigate each PCs, we look at the loading scores: 
<img width="2389" height="790" alt="loading1" src="https://github.com/user-attachments/assets/c2885a26-88d3-444a-b86d-d8b00fe789f8" />
<img width="2390" height="790" alt="loading2" src="https://github.com/user-attachments/assets/081ac278-8c72-4ae3-be37-46b845fc6684" />
<img width="989" height="490" alt="loading3" src="https://github.com/user-attachments/assets/55c225ad-32a1-4171-a793-b1df3497af49" />

And name each components accordingly. We end up with the following principal components:  
| Category | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 |
|---|---|---|---|---|---|---|---|
| Sensation Seeking | General Lifestyle | Impulsivity | Fear Tolerance | Order & Control | Life Stability | Risk Engagement | Novelty Experiment |
| Personality | Social Energy | Emotional Stability | Independence | Behavioral Discipline | Interpersonal & Altruistic | Artistic Interest | — |
| Movie Experience | Immersiveness | Attentiveness | — | — | — | — | — |


## Sensation Seeking and Movie Experience

<img width="300" height="200" alt="sens_vs_movexp" src="https://github.com/user-attachments/assets/27747a63-0b86-43c9-bec4-4f4b2e0ebdb1" />

Above correlation heatmap shows high correlation between life stability and movie immersiveness, but for others indicate low correlations.
To answer the question of whether sensation seeking is related to movie experience, I used pearson's correlation coefficient test with the following hypothesis: 
- H0: Sensation_PC[i] is not related to Movie Experience.
- H1: Higher Sensation_PC[i] is associated with higher Movie Experience.  
We reject the null hypothesis at the α = 0.005 significance level.


**Result**
| Sensation PC       | Movie PC            | **r**     | **p-value**  |
| ------------------ | ------------------- | --------- | ------------ |
| Life Stability     | Movie Immersiveness | **0.134** | **0.000008** |
| Order and Control  | Movie Immersiveness | -0.120    | **0.000067** |
| Impulsivity        | Movie Immersiveness | -0.118    | **0.000086** |
| Fear Tolerance     | Movie Immersiveness | -0.108    | **0.000350** |
| Novelty Experiment | Movie Attentiveness | -0.059    | 0.050651     |
| Novelty Experiment | Movie Immersiveness | -0.046    | 0.124738     |
| Life Stability     | Movie Attentiveness | -0.046    | 0.127477     |
| Impulsivity        | Movie Attentiveness | 0.045     | 0.140126     |
| Fear Tolerance     | Movie Attentiveness | -0.042    | 0.159789     |
| Risk Engagement    | Movie Attentiveness | -0.039    | 0.193391     |
| General Lifestyle  | Movie Attentiveness | -0.026    | 0.392134     |
| Risk Engagement    | Movie Immersiveness | -0.019    | 0.521686     |
| General Lifestyle  | Movie Immersiveness | 0.009     | 0.771970     |
| Order and Control  | Movie Attentiveness | 0.008     | 0.779509     |

Based on the test result, we can conclude: 

- for Life Stability, p-value is statistically significant at alpha = 0.005, and we reject the H0 that there is no relationship between Life Stability and Movie Imemrsiveness. Life stability is positively associated with movie immersiveness. 

- for Order & Control, Impulsivity, and Fear Tolerance, p-value is statistically significant at alpha = 0.005, and we reject the H0 that there is no relationship between these variables and Movie Immersiveness. These variables are negatively associated with movie immersiveness. 

## Personality Types 
To answer the question of whether there is a personality type based on the data of research participants, we use K-Means Clustering and PCA. 
I used the 6 principle components on personality from the above and conducted K-Means Clustering. Using Silhouette Analysis, **k=2** was selected as the optimal number of k.  
For these 2 clusters, each cluster center represents:\
  (i) relatively low social energy & neutral emotional stability\
  (ii) relatively high social energy and neutral emotional stability

<img width="400" height="300" alt="Cluster" src="https://github.com/user-attachments/assets/45ea94e8-4594-4275-8ab3-ff787acc0fb4" />

## Movie Ratings and Viewer Characteristics
To examine rating differences between demographic groups or movie types, I used the Mann-Whitney U Test, which compares two independent distributions without requiring normality—well suited for real-world rating data.

<img width="600" height="500" alt="test_results" src="https://github.com/user-attachments/assets/aa0aea8a-42fa-4185-8183-02fbddd9def8" />

Below is a summary of the test results: 

| Question / Movie Comparison                        | Null Hypothesis (H₀)                                      | Alternative (H₁)                       | Result                | Conclusion                                                      |
| -------------------------------------------------- | --------------------------------------------------------- | -------------------------------------- | --------------------- | --------------------------------------------------------------- |
| Popular vs Niche movies                        | Popular movies are **not rated higher** than niche movies | Popular movies have **higher ratings** | **Reject H₀**         | Popular movies are rated significantly higher than niche movies |
| Shrek (2001): Gender                           | No rating difference between **female and male** viewers  | Ratings differ between genders         | **Fail to reject H₀** | No evidence of gender-based rating difference                   |
| The Lion King (1994): Only-child vs Siblings   | Only-child viewers rate **≤ sibling viewers**             | Only-child viewers rate **higher**     | **Fail to reject H₀** | No evidence only-children enjoy it more                         |
| The Wolf of Wall Street (2013): Social vs Solo | Social viewers rate ≤ solo viewers                        | Social viewers rate higher             | **Fail to reject H₀** | Preference to watch socially does not indicate higher rating    |


## Franchise Movie Ratings Consistency
To answer whether franchise movies have consistent ratings across all titles, I used Kruskal-Wallis Test to determine if the median is consistent across different numberings for each franchise (difference in mean ranks). Kruskal-Wallis is a non-parametric alternative to ANOVA that tests whether multiple independent groups come from the same distribution, making it appropriate here because movie ratings are ordinal (not normally distributed) and we are comparing more than two related groups per franchise.

| Franchise                    | H Statistic | p-value  | # Movies Tested | Conclusion                                     |
| ---------------------------- | ----------- | -------- | --------------- | ---------------------------------------------- |
| **Star Wars**                | 230.584     | 8.02e-48 | 6               | Reject H₀ → Ratings are not consistent    |
| **Harry Potter**             | 3.331       | 0.343    | 4               | **Fail to reject H₀ → Ratings are consistent** |
| **The Matrix**               | 48.379      | 3.12e-11 | 3               | Reject H₀ → Ratings are not consistent     |
| **Indiana Jones**            | 45.794      | 6.27e-10 | 4               | Reject H₀ → Ratings are not consistent     |
| **Jurassic Park**            | 46.591      | 7.64e-11 | 3               | Reject H₀ → Ratings are not consistent     |
| **Pirates of the Caribbean** | 20.644      | 3.29e-05 | 3               | Reject H₀ → Ratings are not consistent     |
| **Toy Story**                | 24.386      | 5.07e-06 | 3               | Reject H₀ → Ratings are not consistent     |
| **Batman**                   | 190.535     | 4.23e-42 | 3               | Reject H₀ → Ratings are not consistent     |

Based on the test, we can see that all franchises except for Star Wars have inconsistent ratings.

<img width="600" height="500" alt="franchise" src="https://github.com/user-attachments/assets/72913976-3ad8-45dc-ad7f-429f5851b70a" />

## Prediction Model: Decision Tree Regressor with 5-fold cross-validation and GridSearchCV

For this task, I built a Decision Tree Regressor to predict movie ratings using personality PCA components as input features. Decision trees are non-linear models that repeatedly split the data into sub-groups based on feature values, making them interpretable and capable of capturing complex interactions.

To prevent overfitting and select the best configuration, I applied GridSearchCV with 5-fold cross-validation, testing multiple values for max_depth, min_samples_leaf, and min_samples_split. This allowed the model to be trained and validated across multiple subsets of the data, ensuring more robust performance. The best model was then retrained on the full training set and evaluated on a hold-out test set. 

The best model achieved RMSE of 0.6938, indicating moderate predictive performance. Feature importance analysis showed that Artistic Interest and Social Energy were the strongest predictors of movie preferences among all personality traits.


## Next Steps
Overall, the analysis shows weak relationships between the variables, and the prediction model performance is relatively weak. This could be caused by:

1. imputation method - maybe just remove nans would be better and cleaner
2. PCA: may need revision
3. some of the viewer characteristic variables are binary (yes/no) questions instead of scale. maybe changing the response to binary values would better explain the variables
4. prediction model - need to look closer and determine which model produces the best performance. Should try multiple different models




