## Covid-19 Infection Prediction
This project aims to predict if an individual is infected with Covid-19 or not.
### Problem Statement
Using historical Covid-19 cases to train machine learning algorithm, the algorithm can then predict if a patient is infected or not.
### Objectives
1. Provide the summary statistics of the variables in the dataset
2. Visualize the relationship among all features using a scatterplot matrix.
3. Train models(logistic regression, random forest classifier, decision tree classifier) on the data.
5. Evaluate the model performance by the evaluation metrics. 
### Project Structure
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    ├── src/
    │   ├── data_cleaning.py
    │   ├── model_training_and_evaluation.py
    │   └── data_exploration.py
    ├── models/
    ├── reports/
    │   └── figures/
    ├── requirements.txt
    ├── README.md
    └── .gitignore

Installation
### Clone repository
git clone git@github.com:SamsonO88/predicting-covid19-infection.git

### Navigate into project
cd predicting-covid19-infection

### Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### Install dependencies
pip install -r requirements.txt

### Evaluation Metrics
Metrics used: 
    1) F1 Score
    2) ROC-AUC
### Technologies Used
    1) Python
    2) Pandas, NumPy
    3) Scikit-learn
    4) Matplotlib / Seaborn
    5) Jupyter Notebook
    6) shap
    7) xgboost

### Insights from Exploratory Data Analysis

1. Most of the male and female population are around the age of 40. In addition to that, as compared by the Covid result, most male that have positive Covid cases are around 40 years of age. This is also the same for females.

2. Generally, male has the highest population and also they possess highest population in terms of positive Covid result from Covid, as compared to female.

3. The heatmap indicate that there are 504 positive Covid cases, where 312 of those cases were men, and 192 were female.

4. Furthermore on the heatmap, 1320 negative Covid cases which is made up of 745 males and 575 females.

5. The male at the median age of 50 show positive results on Covid test as compared to female at the median age of 40.

### License
    MIT License
