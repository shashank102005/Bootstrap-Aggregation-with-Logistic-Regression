
Bootstrap Aggregation with Logistic-Regression

This predictive model was developed in order to predict the result of the breath screening test for detecting the blood alcohol levels , in the people who were suspected to be driving under the influence of alcohol. Bootstrap averaging was used on the Logistic Regression model, to build the predictive model, as the randomness introduced was expected to improve the overall performance of the model.  

HANDLING OF THE CLASS IMBALANCE:

The data was unbalanced , as the number of negatives far outscored the number of posives. To cater to this phenomenon, two different approaches were tested.One was using the weighted logistic regression , and assigning more weight to the rarer class. Second was to oversample the rarer class, in order to compensate for the imbalance.A technique called SMOTE was used for this purpose , which uses the K nearest neighbors method to artificially generate the instances of the rarer class.  The performance of the two methods were evaluated using different metrics such as Area under the ROC curve(AUC) , and the Kolmogorov-Smirnov Point in the K-S curve. 


