##Testing catboost decision trees on the receptor kinase evolution data
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from catboost.utils import get_roc_curve, select_threshold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from skopt.space import Integer, Real
from sklearn.metrics import RocCurveDisplay
from skopt import gp_minimize
from Optimize_catboost import perform_search
import matplotlib.pyplot as plt





####import data#####
df_train = pd.read_csv('RKTrainSet.csv')
#df_predict = pd.read_csv('RKTestSet.csv')
print("Loading data")

#Need to change the output to numbers so it'll work fine. Catboost natively does this for binary, but I have it
#setup here for multiclass if wanted
target_dict = {'Non-Defense':0, 'Defense':1,'Both':2}




##get data and split for training set
features = df_train.drop(["Gene", "Most similar gene", "Category"], axis=1)
target = df_train["Category"]

#print(len(target))




#mlb = MultiLabelBinarizer()
#y_binarized = mlb.fit_transform()

#print(y_binarized)

#exit()


print("Splitting train/test")
#split data
xtrain, xtest, ytrain, ytest = train_test_split(
    features, 
    target,
    random_state=2023,
    test_size=0.2
)

ytrain = ytrain.map(target_dict)
ytest = ytest.map(target_dict)

categorical_features = xtrain.select_dtypes("str").columns.tolist()





###model hyperparameters and assignment
#best for cv is in the comments


print("Optimizing model")
consistent_params = [("loss_function", "Logloss",),
                ("early_stopping_rounds",50),
                ("eval_metric", "AUC"),
                ("auto_class_weights","Balanced"),
                ("verbose", 100)]


#model optimization flag
optimize_model_flag = True
search_type = "bayesian"

#if optimization is turned on, call the optimization script. Otherwise, use manual 
if optimize_model_flag == True:
    model_params = perform_search(xtrain, ytrain, CatBoostClassifier(**dict(consistent_params)), search_type)
    model_params.update(consistent_params)
    model = CatBoostClassifier(**model_params)

else: 
    model = CatBoostClassifier(
        #bagging_temperature=0.3122,
        depth=10, #best is 10, cv 2
        l2_leaf_reg=5, #5, 35 
        random_strength=210, #210, 5
        learning_rate=0.03, #0.03, 0.05
        verbose=100,
        iterations=2000,
        subsample = 0.45,
        loss_function='Logloss',
        early_stopping_rounds=50,
        eval_metric="Accuracy",
        custom_metric=['F1', 'Precision', 'Recall', "BalancedAccuracy"],
        auto_class_weights='Balanced'
    )

    

##cross fold validation. 
#cv_results = cv(
#    pool=Pool(xtest,ytest,cat_features=categorical_features),
#    params=model.get_params(),
#    fold_count=10,
#    shuffle=True,
#    stratified=True
#)

#print(cv_results.tail(1))
#best_accuracy = cv_results['test-Accuracy-mean'].max()
#print(f"Best mean validation accuracy: {best_accuracy}")

print("Training model")
#fit the model and output the training accuracy. Then look at testing accuracy 
model.fit(xtrain, ytrain, eval_set=(xtrain,ytrain), cat_features = categorical_features, use_best_model=True)

roc_curve_values = get_roc_curve(model, Pool(xtrain, ytrain, cat_features=categorical_features), plot=False)

#print(roc_curve_values['fpr'])
#print(roc_curve_values['tpr'])
#print(roc_curve_values['thresholds'])

target_fpr = 0.01
optimal_threshold = select_threshold(model, curve=roc_curve_values)
print(f"Optimal Threshold for {target_fpr*100}% FPR: {optimal_threshold}")

model.set_probability_threshold(optimal_threshold)
ypred = model.predict(xtest)   #get prediction
yprob = model.predict_proba(xtest)[:,1]  #get probability
#yprob = (yprob > optimal_threshold).astype(int)


accuracy = accuracy_score(ytest,ypred)
#accuracy = model.score(xtest,ytest)
f1score = f1_score(ytest, ypred, average='weighted')
roc_auc = roc_auc_score(ytest, yprob)
#print('Classification report:')
#print(classification_report(ytest, ypred))

#print(model.get_best_score()['validation'])
#print(f"ROC AUC Score: {roc_auc:.4f}")

print(f"Accuracy: {accuracy}", f"f1: {f1score}", f"ROC AUC: {roc_auc:.4f}")
###
# print(f"Accuracy: {model.get_best_score()['validation']['Accuracy']}", 
#       f"f1: {model.get_best_score()['validation']['F1']}", 
#       f"Precision: {model.get_best_score()['validation']['Precision']}",
#       f"Recall: {model.get_best_score()['validation']['Recall']}")
# ###


#RocCurveDisplay.from_predictions(model, ytest, ypred)
#plt.title('Received Operating Characeristic')
#plt.show()
# Save the model in the default CBM format
#model.save_model('CatboostModel_RKs.cbm') 
