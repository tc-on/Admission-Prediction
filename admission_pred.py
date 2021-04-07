#!/usr/bin/env python
# coding: utf-8

# <a id="lib"></a>
# # 1. Import Libraries

# In[1]:


# import 'Pandas' 
import pandas as pd 

# import 'Numpy' 
import numpy as np

# import subpackage of Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# import 'Seaborn' 
import seaborn as sns

# suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# display all columns of the dataframe
pd.options.display.max_columns = None

# display all rows of the dataframe
pd.options.display.max_rows = None
 
# display the float values upto 6 decimal places     
pd.options.display.float_format = '{:.6f}'.format

# import train-test split 
from sklearn.model_selection import train_test_split

# import StandardScaler to perform scaling
from sklearn.preprocessing import StandardScaler 

# import various functions from sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# import the XGBoost function for classification
from xgboost import XGBClassifier


# In[2]:


# plot size using 'rcParams'
plt.rcParams['figure.figsize'] = [15,8]


# <a id="prep"></a>
# # 2. Data Preparation

# <a id="read"></a>
# ## 2.1 Read the Data

# In[3]:


# load the csv file
df_admissions = pd.read_csv('Admission_predict.csv')

# display first five observations using head()
df_admissions.head()


# In[4]:


# 'shape' to check the dimensions
df_admissions.shape


# **Interpretation:** The data has 400 observations and 9 variables.

# <a id="dtype"></a>
# ## 2.2 Check the Data Type

# In[5]:


# using 'dtypes' to check the data type of a variable
df_admissions.dtypes


# **Interpretation:** The variables `GRE Score`, `TOEFL Score`, `University Rating`, `SOP`, `LOR` and `CGPA` are numerical.
# 
# From the above output, we see that the data type of `Research` is 'int64'.
# 
# But according to the data definition, `Research` is a categorical variable, which is wrongly interpreted as 'int64', so we will convert these variables data type to 'object'.

# In[6]:


# convert numerical variable to categorical (object) 
df_admissions['Research'] = df_admissions['Research'].astype(object)


# In[7]:


# recheck the data types using 'dtypes'
df_admissions.dtypes


# **Interpretation:** Now, all the variables have the correct data type.

# <a id="drop"></a>
# ## 2.3 Remove Insignificant Variables

# In[8]:


# drop the column 'Serial No.' using drop()
df_admissions = df_admissions.drop('Serial No.', axis = 1)


# <a id="dist"></a>
# ## 2.4 Distribution of Variables

# **Distribution of numeric independent variables.**

# In[9]:


# for the independent numeric variables, plot the histogram to check the distribution of the variables
df_admissions.drop('Chance of Admit', axis = 1).hist()

# adjust the subplots
plt.tight_layout()

# display the plot
plt.show()  

# print the skewness for each numeric independent variable
print('Skewness:')

# drop the target variable using drop()
df_admissions.drop('Chance of Admit', axis = 1).skew()


# **Interpretation:** The above plot indicates that all the variables are near normally distributed.

# **Distribution of categoric independent variable.**

# In[10]:


# for the independent categoric variable, plot the count plot to check the distribution of the variable 'Research'
sns.countplot(df_admissions.Research)

# plot and axes labels
plt.title('Count Plot for Categorical Variable (Research)', fontsize = 15)
plt.xlabel('Research', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

# display the plot
plt.show()


# #### Distribution of dependent variable.

# In[11]:


# only the target variable
df_target = df_admissions['Chance of Admit'].copy()

# counts of 0's and 1's in the 'Chance of Admit' variable
df_target.value_counts()

# plot the countplot of the variable 'Chance of Admit'
sns.countplot(x = df_target)

# code to print the values in the graph
plt.text(x = -0.05, y = df_target.value_counts()[0] + 1, s = str(round((df_target.value_counts()[0])*100/len(df_target),2)) + '%')
plt.text(x = 0.95, y = df_target.value_counts()[1] +1, s = str(round((df_target.value_counts()[1])*100/len(df_target),2)) + '%')

# plot and axes labels
plt.title('Count Plot for Target Variable (Chance of Admit)', fontsize = 15)
plt.xlabel('Target Variable', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

# to show the plot
plt.show()


# **Interpretation:** The above plot shows that there is no imbalance in the target variable.

# <a id="null"></a>
# ## 2.5 Missing Value Treatment

# In[12]:


# sorting the variables on the basis of total null values in the variable
Total = df_admissions.isnull().sum().sort_values(ascending=False)          

# calculating percentage of missing values
Percent = (df_admissions.isnull().sum()*100/df_admissions.isnull().count()).sort_values(ascending=False)   

# concatenating the 'Total' and 'Percent' columns using 'concat' function
missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])    
missing_data


# **Interpretation:** The above output shows that there are no missing values in the data.

# <a id="dummy"></a>
# ## 2.6 Dummy Encode the Categorical Variables

# In[13]:


# storing the target variable 'Chance of Admit' in a dataframe 'df_target'
df_target = df_admissions['Chance of Admit']

# storing all the independent variables in a dataframe 'df_feature' 
df_feature = df_admissions.drop('Chance of Admit', axis = 1)


# In[14]:


# filtering the numerical features in the dataset
df_num = df_feature.select_dtypes(include = [np.number])

# displaying numerical features
df_num.columns


# In[15]:


# filtering the categorical features in the dataset
df_cat = df_feature.select_dtypes(include = [np.object])

# displaying categorical features
df_cat.columns


# In[16]:


# using 'get_dummies' from pandas to create dummy variables
dummy_var = pd.get_dummies(data = df_cat, drop_first = True)


# In[17]:


# concat the dummy variables with numeric features to create a dataframe of all independent variables
X = pd.concat([df_num, dummy_var], axis = 1)

# display first five observations
X.head()


# <a id="split"></a>
# ## 2.7 Train-Test Split

# In[18]:


# split data into train subset and test subset
X_train, X_test, y_train, y_test = train_test_split(X, df_target, random_state = 10, test_size = 0.2)

# check the dimensions of the train & test subset using 'shape'
print('X_train', X_train.shape)
print('y_train', y_train.shape)

# print dimension of test set
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# #### A generalized function to calculate the metrics for the test set.

# In[19]:


# create a generalized function to calculate the metrics values for test set
def get_test_report(model):
    
    # for test set:
    test_pred = model.predict(X_test)

    # return the classification report for test data
    return(classification_report(y_test, test_pred))


# #### Plotting the confusion matrix.

# In[20]:


def plot_confusion_matrix(model):
    y_pred = model.predict(X_test)
    
    # create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # label the confusion matrix  '
    conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])

    # plot a heatmap to visualize the confusion matrix
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, 
                linewidths = 0.1, annot_kws = {'size':25})

    # set the font size of x-axis ticks using 'fontsize'
    plt.xticks(fontsize = 20)

    # set the font size of y-axis ticks using 'fontsize'
    plt.yticks(fontsize = 20)

    # display the plot
    plt.show()


# #### Plotting the ROC curve.

# In[21]:


def plot_roc(model):
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    # the roc_curve() returns the values for false positive rate, true positive rate and threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # plot the ROC curve
    plt.plot(fpr, tpr)

    # set limits for x and y axes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # plot the straight line showing worst prediction for the model
    plt.plot([0, 1], [0, 1],'r--')

    # add plot and axes labels
    plt.title('ROC curve for Admission Prediction Classifier', fontsize = 15)
    plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)
    plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)

    # add the AUC score to the plot
    plt.text(x = 0.82, y = 0.3, s = ('AUC Score:',round(roc_auc_score(y_test, y_pred_prob),4)))

    # plot the grid
    plt.grid(True)


# <a id="boosting"></a>
# # 3. Boosting Methods

# <a id="ada"></a>
# ## 3.1 AdaBoost

# In[22]:


# instantiate the 'AdaBoostClassifier'
ada_model = AdaBoostClassifier(n_estimators = 40, random_state = 10)

# fit the model using fit() on train data
ada_model.fit(X_train, y_train)


# Let us understand the parameters in the `AdaBoostClassifier()`:
# 
# `algorithm=SAMME.R`: It is the default boosting algorithm. This algorithm uses predicted class probabilities to build the stumps.
# 
# `base_estimator=None`: By default, the estimator is a decision tree with a maximum depth equal to 1 (stump).
# 
# `learning_rate=1.0`: It considers the contribution of each estimator in the classifier.
# 
# `n_estimators=40`: It is the number of estimators at which boosting is terminated.
# 
# `random_state=10`: It returns the same set of samples for each code implementation.

# #### Plotting the confusion matrix.

# In[23]:


# call the function to plot the confusion matrix
plot_confusion_matrix(ada_model)


# **Calculating performance measures on the test set.**

# In[24]:


# compute the performance measures on test data
test_report = get_test_report(ada_model)

# print the performance measures
print(test_report)


# **Interpretation:** The output shows that the model is 81% accurate.

# #### Plotting the ROC curve.

# In[25]:


# call the function to plot the ROC curve
plot_roc(ada_model)


# **Interpretation:** The red dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).<br>
# From the above plot, we can see that the AdaBoost model is away from the dotted line; with the AUC score 0.9132.

# <a id="gradient"></a>
# ## 3.2 Gradient Boosting

# In[26]:


# instantiate the 'GradientBoostingClassifier' 
gboost_model = GradientBoostingClassifier(n_estimators = 150, max_depth = 10, random_state = 10)

# fit the model using fit() on train data
gboost_model.fit(X_train, y_train)


# Let us understand the parameters in the `GradientBoostingClassifier()`:
# 
# `ccp_alpha=0.0`: The complexity parameter used for pruning. By default, there is no pruning.
# 
# `criterion=friedman_mse`: The criteria to measure the quality of a split.
# 
# `init=None`: The estimator for initial predictions.
# 
# `learning_rate=0.1`: It considers the contribution of each estimator in the classifier.
# 
# `loss=deviance`: The loss function to be optimized.
# 
# `max_depth=10`: Assigns the maximum depth of the tree.
# 
# `max_features=None`: Maximum features to consider for the split.
# 
# `max_leaf_nodes=None`: Maximum number of leaf/terminal nodes in the tree.
# 
# `min_impurity_decrease=0.0`: A node splits if it decreases the impurity by the value given by this parameter. 
# 
# `min_impurity_split=None`: Minimum value of impurity for a node to split.
# 
# `min_samples_leaf=1`: Minimum number of samples needed at the leaf/terminal node.
# 
# `min_samples_split=2`: Minimum number of samples needed at the internal node to split. 
# 
# `min_weight_fraction_leaf=0.0`: Minimum weighted fraction needed at a leaf node.
# 
# `n_estimators=150`: The number of estimators to consider.
# 
# `n_iter_no_change=None`: Number of iterations after which the training should terminate if the score is not improving.
# 
# `presort='deprecated'`: It considers whether to presort the data. (This parameter may not be available in the latest versions). 
# 
# `random_state=10`: It returns the same set of samples for each code implementation.
# 
# `subsample=1.0`: Fraction of samples to use for fitting each estimator.
# 
# `tol=0.0001`: Value of tolerance to terminate the training.
# 
# `validation_fraction=0.1`: Fraction of training dataset used for validation.
# 
# `verbose=0`: Enables verbose output (by default, no progress will be printed).
# 
# `warm_start=False`: Whether to reuse the solution of previous code implementation (by default, it does not consider the previous solution).

# #### Plotting the confusion matrix.

# In[27]:


# call the function to plot the confusion matrix
plot_confusion_matrix(gboost_model)


# **Calculating performance measures on the test set.**

# In[28]:


# compute the performance measures on test data
test_report = get_test_report(gboost_model)

# print the performance measures
print(test_report)


# **Interpretation:** The classification report shows that the model is 79% accurate. Also, the sensitivity and specificity are equal.

# #### Plotting the ROC curve.

# In[29]:


# call the function to plot the ROC curve
plot_roc(gboost_model)


# **Interpretation:** The red dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).<br>
# From the above plot, we can see that the gradient boosting model is away from the dotted line; with the AUC score 0.8954.

# <a id="xgboost"></a>
# ## 3.3 XGBoost 

# In[30]:


# instantiate the 'XGBClassifier'
xgb_model = XGBClassifier(max_depth = 10, gamma = 1)

# fit the model using fit() on train data
xgb_model.fit(X_train, y_train)


# Let us understand the parameters in the `XGBClassifier()`:
# 
# `base_score=0.5`: Initial prediction for base learners.
# 
# `booster=gbtree`: Considers the regression tree as the base learners.
# 
# `colsample_bylevel=1`: Fraction of variables to consider for each level.
# 
# `colsample_bynode=1`: Fraction of variables to consider for each split.
# 
# `colsample_bytree=1`: Fraction of variables to consider for each tree.
# 
# `gamma=1`: Value of minimum loss reduction required for the partition of the leaf node.
# 
# `gpu_id=-1`: It considers all the GPU's. 
# 
# `importance_type=gain`: Importance type for calculating feature importance.
# 
# `interaction_constraints=''`: By default, no interaction between the features is allowed.
# 
# `learning_rate=0.300000012`: It considers the contribution of each estimator in the classifier.
# 
# `max_delta_step=0`: Maximum delta step allowed for each tree's weight estimation to be.
# 
# `max_depth=10`: Maximum depth of each tree.
# 
# `min_child_weight=1`: Minimum sum of hessian (p*(1-p)) required in a leaf node.
# 
# `missing=nan`: Value to consider as a missing value.
# 
# `monotone_constraints='()'`:  Constraint of variable monotonicity. (adding increasing/decreasing constraint on the variables )
# 
# `n_estimators=100`: The number of estimators to consider.
# 
# `n_jobs=0`: Number of parallel threads to run the classifier.
# 
# `num_parallel_tree=1`: It is used for boosting random forest.
# 
# `objective='binary:logistic'`: Considers the binary logistic regression as a learning objective.
# 
# `random_state=0`: It returns the same set of samples for each code implementation.
# 
# `reg_alpha=0`: Lasso regularization term for weights.
# 
# `reg_lambda=1`: Ridge regularization term for weights.
# 
# `scale_pos_weight=1`:  Ratio of the number of negative class to the positive class.
# 
# `subsample=1`: Fraction of total training data points.
# 
# `tree_method='exact'`: Considers the exact greedy algorithm.
# 
# `validate_parameters=1`: Performs validation on input paramerters.
# 
# `verbosity=None`: Enables verbose output (by default, no progress will be printed).

# #### Plotting the confusion matrix.

# In[31]:


# call the function to plot the confusion matrix
plot_confusion_matrix(xgb_model)


# **Calculating performance measures on the test set.**

# In[32]:


# compute the performance measures on test data
test_report = get_test_report(xgb_model)

# print the performance measures
print(test_report)


# **Interpretation:** The above output shows that the f1-score and accuracy of the model is 0.84

# #### Plotting the ROC curve.

# In[33]:


# pass the XGBoost model to the function
plot_roc(xgb_model)


# **Interpretation:** The red dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).<br>
# From the above plot, we can see that the XGBoost model is away from the dotted line; with the AUC score 0.8888.

# <a id="tuning"></a>
# ### 3.3.1 Tuning the Hyperparameters (GridSearchCV)

# In[34]:


# create a dictionary with hyperparameters and its values
tuning_parameters = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                     'max_depth': range(3,10),
                     'gamma': [0, 1, 2, 3, 4]}

# instantiate the 'XGBClassifier' 
xgb_model = XGBClassifier()

# use GridSearchCV() to find the optimal value of the hyperparameters
xgb_grid = GridSearchCV(estimator = xgb_model, param_grid = tuning_parameters, cv = 3, scoring = 'roc_auc')

# fit the model on X_train and y_train using fit()
xgb_grid.fit(X_train, y_train)

# get the best parameters
print('Best parameters for XGBoost classifier: ', xgb_grid.best_params_, '\n')


# #### Building the model using the tuned hyperparameters.

# In[35]:


# instantiate the 'XGBClassifier'
xgb_grid_model = XGBClassifier(learning_rate = xgb_grid.best_params_.get('learning_rate'),
                               max_depth = xgb_grid.best_params_.get('max_depth'),
                              gamma = xgb_grid.best_params_.get('gamma'))

# use fit() to fit the model on the train set
xgb_model = xgb_grid_model.fit(X_train, y_train)

# print the performance measures for test set for the model with best parameters
print('Classification Report for test set:\n', get_test_report(xgb_model))


# **Interpretation:** The above output shows that the f1-score and accuracy of the model is 0.84.

# #### Plotting the ROC curve.

# In[36]:


# call the function to plot the ROC curve
plot_roc(xgb_model)


# **Interpretation:** The red dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).<br>
# From the above plot, we can see that the XGBoost model (GridSearchCV) is away from the dotted line; with the AUC score 0.9145.

# ### Identifying the Important Features using XGBoost

# In[37]:


# create a dataframe that stores the feature names and their importance
important_features = pd.DataFrame({'Features': X_train.columns, 
                                   'Importance': xgb_model.feature_importances_})

# sort the dataframe in the descending order according to the feature importance
important_features = important_features.sort_values('Importance', ascending = False)

# create a barplot to visualize the features based on their importance
sns.barplot(x = 'Importance', y = 'Features', data = important_features)

# add plot and axes labels
plt.title('Feature Importance', fontsize = 15)
plt.xlabel('Importance', fontsize = 15)
plt.ylabel('Features', fontsize = 15)

# display the plot
plt.show()


# **Interpretation:** The above bar plot shows that, the variable `CGPA` is of highest importance. 

# <a id="stack"></a>
# # 4. Stack Generalization  

# In[38]:


# initialize the standard scalar
X_scaler = StandardScaler()

# scale all the numerical columns
num_scaled = X_scaler.fit_transform(df_num)

# create a dataframe of scaled numerical variables
df_num_scaled = pd.DataFrame(num_scaled, columns = df_num.columns)


# Concatenate scaled numerical and dummy encoded categorical variables.

# In[39]:


# concat the dummy variables with scaled numeric features to create a dataframe of all independent variables
X = pd.concat([df_num_scaled, dummy_var], axis = 1)

# display first five observations
X.head()


# Let us split the dataset in train and test set.

# In[40]:


# split data into train subset and test subset
X_train, X_test, y_train, y_test = train_test_split(X, df_target, random_state = 10, test_size = 0.2)

# check the dimensions of the train & test subset using 'shape'
print('X_train', X_train.shape)
print('y_train', y_train.shape)

# print dimension of test set
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# #### Stacking Classifier using the Random forest, KNN and Naive bayes as base learners.

# In[41]:


# consider the various algorithms as base learners
base_learners = [('rf_model', RandomForestClassifier(criterion = 'entropy', max_depth = 10, max_features = 'sqrt', 
                                                     max_leaf_nodes = 8, min_samples_leaf = 5, min_samples_split = 2, 
                                                     n_estimators = 50, random_state = 10)),
                 ('KNN_model', KNeighborsClassifier(n_neighbors = 17, metric = 'euclidean')),
                 ('NB_model', GaussianNB())]

# initialize stacking classifier 
stack_model = StackingClassifier(estimators = base_learners, final_estimator = GaussianNB())

# fit the model on train dataset
stack_model.fit(X_train, y_train)


# #### Plotting the confusion matrix.

# In[42]:


# call the function to plot the confusion matrix
plot_confusion_matrix(stack_model)


# **Calculating performance measures on the test set.**

# In[43]:


# compute the performance measures on test data
test_report = get_test_report(stack_model)

# print the performance measures
print(test_report)


# **Interpretation:** The above output shows that the f1-score and accuracy of the model is 0.86

# #### Plotting the ROC curve.

# In[44]:


# call the function to plot the ROC curve
plot_roc(stack_model)


# **Interpretation:** The red dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).<br>
# From the above plot, we can see that the stacking model is away from the dotted line; with the AUC score 0.9492.
