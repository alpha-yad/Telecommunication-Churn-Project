
# Predicting Customer Churn in a Telecommunications Company


Predicting customer churn is a critical task for telecommunications companies aiming to maintain a loyal customer base. By leveraging historical data and advanced machine learning techniques, companies can not only predict churn but also implement effective retention strategies to minimize it.

## Importing Dependencies


```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

    
## Collecting data
I have used this telecom service customer churn dataset for this particular project- [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
## Data Preprocessing
For the data processing part, I first converted 'TotalCharges' from an object (string) to a float type, as 'TotalCharges' is important in deciding customer churn. Next, I encoded the categorical data to a binary or machine-readable format. I also dropped the 'customerID' column, which is a string type and could create an imbalance in our dataset. In final step Divides the data into training and testing sets here our dependent variable is set to churn column.
## Visualization
For visualization, I created tenure bins to understand the records of fresh users and established users churning in the future. I also performed demographic analysis on gender, partner, senior citizen, and dependents to examine the nature of users. I found that most customers in this dataset are younger individuals without dependents, and there is a relatively equal distribution of user gender and marital status.

To show the relationship between cost and customer churn, I used a Plotly violin plot. The plot indicates that customers who churn have a higher median monthly charge. Additionally, I analyzed the relationship between customer churn and a few other categorical variables. I discovered that customers using fiber optic services churn more often than others, those without tech support churn more frequently, and many who churned did not have online backup. Furthermore, users with monthly contracts are more likely to churn compared to those with longer-term contracts.
## Building model

For building the model, I used a Random Forest and a Support Vector Machine (SVM) and combined their predictions using a Voting Classifier.
## prediction result

To evaluate the performance of the model, I will use accuracy score

```bash
 Accuracy: 0.7341862117981521
```
The accuracy of this model is 0.734, indicating our model is performing well
