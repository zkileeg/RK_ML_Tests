# RK_ML_Tests
Testing various ML algorithms and applications on the RK pangenome data to try and predict function from what we have. 


**Catboost model**:

So far, the best I've been able to achieve for model accuracy/f1 score/ROC AUC is:

Accuracy: 0.7164179104477612 f1: 0.7304376991447259 ROC AUC: 0.8073


I'm getting there slowly but surely. So far I've tried:
- Bayesian hyperparameter selection
- random search
- threshold optimization
- weighting for unbalanced sets

I have a few things I think are causing the issues right now. 
1. I'm using selection data. A lot of outputs are 0 even if that gene is disease related. It may be confusing the output. There may also only be a weak association between positive selection and disease relatedness
2. Many genes are dual functional. That is, they are involved in both the positive (disease) and negative (growth) cases. I'm looking at binary classification to start, but it may be a good idea to also add a "both" class and see how that works. 
