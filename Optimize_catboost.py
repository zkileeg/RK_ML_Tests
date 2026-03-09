##Testing catboost decision trees on the receptor kinase evolution data
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from skopt.space import Integer, Real, randint, uniform
from skopt import BayesSearchCV





####import data#####
#df_train = pd.read_csv('RKTrainSet.csv')
#df_predict = pd.read_csv('RKTestSet.csv')


#Need to change the output to numbers so it'll work fine. Catboost natively does this for binary, but I have it
#setup here for multiclass if wanted
#target_dict = {'Non-Defense':0, 'Defense':1,'Both':2}




##get data and split for training set
#features = df_train.drop(["Gene", "Most similar gene", "Category"], axis=1)
#target = df_train["Category"]

#print(len(target))




#mlb = MultiLabelBinarizer()
#y_binarized = mlb.fit_transform()

#print(y_binarized)

#exit()



#split data
#xtrain, xtest, ytrain, ytest = train_test_split(
#    features, 
#    target,
#    random_state=2023,
#    test_size=0.2
#)

#ytrain = ytrain.map(target_dict)
#ytest = ytest.map(target_dict)

        

        
        
        
        


###model hyperparameters and assignment
#best for cv is in the comments

#model = CatBoostClassifier(
#    verbose=0,
#    loss_function="Logloss",
#    early_stopping_rounds=50,
#    eval_metric="AUC",
#    auto_class_weights="Balanced"
#)

        

        
        
        

#exit()
def perform_search(xtrain, ytrain, model, search_method):

    print(search_method)

    if search_method != 'bayesian' and search_method != 'random':
        print("Error: Optimization search type is not correct. Options are bayesian or random")
        exit()
      

    categorical_features = xtrain.select_dtypes("str").columns.tolist()

    #search_space = {
    #    'depth': Integer(14,15),
    #    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    #    'iterations': Integer(1,2),
    #    'l2_leaf_reg': Real(80,100,'uniform'),
    #    'random_strength': Real(299,300, 'log-uniform'),
    #    'bagging_temperature': Real(0,1,'uniform')

     #   }

    

    if not categorical_features:
        cat_indices = ""
        print(cat_indices)
            
    else:
        cat_indices = [xtrain.columns.get_loc(col) for col in categorical_features]
        print(cat_indices)



    
    

    match search_method:

        case 'bayesian': 

            #define search space for bayeisan. 
            search_space = {
            'depth': (2, 10),
            'learning_rate': (0.01, 0.1, 'log-uniform'),
            'l2_leaf_reg': (1, 10, 'log-uniform'),
            'iterations': Integer(1,500),
            'random_strength': (1, 300, 'log-uniform'),
            'bagging_temperature': (0,1,'uniform'),
            'subsample': (0.2,0.6, 'uniform')
                
                }
           
            
            search_params = BayesSearchCV(
                    estimator=model,
                    search_spaces=search_space,
                    n_iter=10,
                    cv=3,
                    random_state=2023,
                    scoring='f1_weighted',
                    fit_params = {'cat_features': cat_indices}
                ) #end bayesian case

            #search_params.fit(xtrain, ytrain, cat_features=cat_indices)

        case 'random': 

            #define search space for random search. 
            search_space = {
                'depth': randint(2, 10),
                'learning_rate': uniform(0.01, 0.1),
                'l2_leaf_reg': loguniform(1, 10),
                'iterations': randint(1,500),
                'random_strength': loguniform(1, 300),
                'bagging_temperature': uniform(0,1),
                'subsample': uniform(0.2,0.6)
                
                }

            search_params = RandomizedSearchCV(
                estimator=model,
                param_distributions=search_space,
                n_iter=20,
                cv=3,
                scoring='f1_macro'
                
            ) #end random case

    #fit parameters to find best model.
    search_params.fit(xtrain, ytrain, cat_features=cat_indices)

# # Report the best result and configuration
    print(f"Best score: {search_params.best_score_}")
    print(f"Best params: {search_params.best_params_}")

    return(search_params.best_params_)
