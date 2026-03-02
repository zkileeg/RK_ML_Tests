##Testing catboost decision trees on the receptor kinase evolution data
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt




####import data#####
df_train = pd.read_csv('RKTrainSet.csv')
df_predict = pd.read_csv('RKTestSet.csv')


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
model = CatBoostClassifier(
    cat_features=categorical_features,
    depth=10, #best is 10, cv 2
    l2_leaf_reg=5, #5, 35 
    random_strength=10, #210, 5
    learning_rate=0.03, #0.03, 0.05
    verbose=100,
    iterations=500,
    loss_function='Logloss',
    early_stopping_rounds=50,
    custom_loss=["Accuracy"],
    eval_metric="Accuracy",
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


#fit the model and output the training accuracy. Then look at testing accuracy 
model.fit(xtrain, ytrain, eval_set=(xtrain,ytrain))

ypred = model.predict(xtest)   #get prediction
yprob = model.predict_proba(xtest)   #get probability

accuracy = accuracy_score(ytest,ypred)
#accuracy = model.score(xtest,ytest)
f1score = f1_score(ytest, ypred, average='weighted')

print(f"Accuracy: {accuracy}", f"f1: {f1score}")


# Save the model in the default CBM format
model.save_model('CatboostModel_RKs.cbm') 
