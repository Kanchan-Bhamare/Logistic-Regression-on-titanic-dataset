# Logistic Regression project on Titanic dataset

## 1. Problem statement :

Problem Statement:
The goal is to **predict survival** of passengers travelling in RMS **Titanic** using **Logistic regression**.
##  2.Data Loading and Description :

![enter image description here](https://github.com/Kanchan-Bhamare/Logistic-Regression-on-titanic-dataset/blob/main/picture-1.png?raw=true)

 -   The dataset consists of the information about people boarding the famous RMS Titanic. Various variables present in the dataset includes data of age, sex, fare, ticket etc.
 -  The dataset comprises of  **891 observations of 12 columns**.
 - 

## Importing packages :

 - numpy
 - pandas
 - matplotlib
 - seaborn
 

## Importing Dataset :
Importing training dataset using pd.read_csv

## Preprocessing the data :
Here we are doing simple exploratory data analysis.
Dealing with missing values

 

 1. Dropping / Replacing missing entries of Embarked.
 2. Replacing missing values of **Age** and **Fare** with median values.
 3. Dropping the column  **'Cabin'**  as it has too many  _null_  values.
 4. Drawing **pair plot** to know the joint relationship between **'Fare' , 'Age' , 'Pclass' & 'Survived'**


![enter image description here](https://github.com/Kanchan-Bhamare/Logistic-Regression-on-titanic-dataset/blob/main/picture-2.png?raw=true)

Observing the diagonal elements,

 1.   More people of  **Pclass 1**  _survived_  than died (First peak of red is higher than blue)
 2.   More people of  **Pclass 3**  _died_  than survived (Third peak of blue is higher than red)
 3.   More people of age group  **20-40 died**  than survived.
 4.   Most of the people paying  **less fare died**.


 5. Establishing **coorelation** between all the features using **heatmap**.

 
![enter image description here](https://github.com/Kanchan-Bhamare/Logistic-Regression-on-titanic-dataset/blob/main/picture-3.png?raw=true)

-   **Age and Pclass are negatively corelated with Survived.**
-   FamilySize is made from Parch and SibSb only therefore high positive corelation among them.
-   **Fare and FamilySize**  are  **positively coorelated with Survived.**
-   With high corelation we face  **redundancy**  issues.

# 4. Logistic Regression :

## 4.1 Introduction to Logistic Regression 

Logistic regression is a techinque used for solving the **classification problem**.  
And Classification is nothing but a problem of **identifing** to which of a set of **categories** a new observation belongs, on the basis of _training dataset_ containing observations (or instances) whose categorical membership is known.  
For example to predict:  
**Whether an email is spam (1) or not (0)** or,  
**Whether the tumor is malignant (1) or not (0)  
**Below is the pictorial representation of a basic logistic regression model to classify set of images into _happy or sad._


![enter image description here](https://github.com/Kanchan-Bhamare/Logistic-Regression-on-titanic-dataset/blob/main/picture-4.png?raw=true)

Both Linear regression and Logistic regression are  **supervised learning techinques**. But for the  _Regression_  problem the output is  **continuous**  unlike the  _classification_  problem where the output is  **discrete**.  

-   Logistic Regression is used when the  **dependent variable(target) is categorical**.  
    
-   **Sigmoid function**  or logistic function is used as  _hypothesis function_  for logistic regression.


## 4.2 Preparing X and y using pandas
X = titanic.loc[:,titanic.columns != 'Survived']
y = titanic.Survived 

## 4.3 Splitting X and y into training and test datasets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

## 4.4 Logistic regression in scikit-learn

To apply any machine learning algorithm on your dataset, basically there are 4 steps:

1.  Load the algorithm
2.  Instantiate and Fit the model to the training dataset
3.  Prediction on the test set
4.  Calculating the accuracy of the model

The code block given below shows how these steps are carried out:  

`from sklearn.linear_model import LogisticRegression logreg = LogisticRegression() logreg.fit(X_train, y_train) accuracy_score(y_test,y_pred_test))`

## 4.5 Using the Model for Prediction

y_pred_train = logreg.predict(X_train)

y_pred_test = logreg.predict(X_test)    

 - We need an evaluation metric in order to compare our predictions with the actual values.


# Model evaluation :

**Error** is the _deviation_ of the values _predicted_ by the model with the _true_ values.  
We will use **accuracy score __ and __confusion matrix** for evaluation.

### 5.1 Model Evaluation using  **accuracy classification score**
from sklearn.metrics import accuracy_score
print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test))

 - Accuracy score for test data is: 0.8044692737430168

### 5.2 Model Evaluation using confusion matrix

A  **confusion matrix**  is a  **summary**  of prediction results on a classification problem.

The number of correct and incorrect predictions are summarized with count values and broken down by each class.  
Below is a diagram showing a general confusion matrix.

![enter image description here](https://github.com/Kanchan-Bhamare/Logistic-Regression-on-titanic-dataset/blob/main/picture-5.png?raw=true)


|Predicted Died    | Predicted  |Survived
|Actual Died    --|-97 -|9
| Actual Survived  | 26    |47


This means 93 + 48 =  **141 correct predictions**  & 25 + 13 =  **38 false predictions**.

**Adjusting Threshold**  for predicting Died or Survived.

 -   In the section we have used,  **.predict**  method for classification. This method takes 0.5 as the default threshold for prediction.  
    
 -   Now, we are going to see the impact of changing threshold on the accuracy of our logistic regression model.  
    
 -   For this we are going to use  **.predict_proba**  method instead of using .predict method.


 -1. Setting the threshold to **0.75**

Accuracy score for test data is: 0.7430167597765364

The accuracy have been  **reduced**  significantly changing from  **0.79 to 0.73**. Hence, 0.75 is  **not a good threshold**  for our model.

 - 2.Setting the threshold to **0.25**

Accuracy score for test data is: 0.7653631284916201

The accuracy have been  **reduced**, changing from  **0.79 to 0.75**. Hence, 0.25 is also  **not a good threshold**  for our model.  
Later on we will see methods to identify the best threshold.
