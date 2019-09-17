# Table of Contents
* [Concepts](#concepts)
  * [What is Machine Learning for?](#what-is-machine-learning-for)
  * [What is Machine Learning?](#what-is-machine-learning)
  * **[How to Setup a Business Problem as a Machine Learning Problem?](#how-to-setup-a-business-problem-as-a-machine-learning-problem)**
  * [Featurization or Feature Extraction](#featurization-or-feature-extraction)
  * [Loss Function](#loss-function)
  * [Main Principle of Train/Test Split](#main-principle-of-traintest-split)
  * [What Might go Wrong with ML?](#what-might-go-wrong-with-ml)
  * [Model Complexity and Overfitting](#model-complexity-and-overfitting)  
* [Cross-Validation](#cross-validation)
  * [What's the Purpose of Cross-Validation?](#whats-the-purpose-of-cross-validation)
  * [Types of Cross-Validation](#types-of-cross-validation)
  * [How to Choose K?](#how-to-choose-k)
  * [The Right Way to Do Cross-Validation](#the-right-way-to-do-cross-validation)
* [](#)
* [](#)



# Concepts
## What is Machine Learning for?
* Solve a prediction problem: given an input `x`, predict an "appropriate" output `y`
  * Binary Classification
  * Multiclass
  * Regression

* Describe how the given data are organized or clustered.
  * Clustering


## What is Machine Learning?

> Prediction function: It takes input `x` and produce an output `y`.

> A machine learning algorithm takes "training data" as input, "learns" from the training data, and generates an ouptut, which is a "predcition function". Machine learning helps to find the **best prediction funciton**. 

* Machine learning is basically programming with data. [Alex Smola, CMU 10701-15]
* Roughly speaking learning is the process of converting experience into expertise of knowledge.

## How to Setup a Business Problem as a Machine Learning Problem?
* What is the business manager really looking for?
* What type of an ML problem is this?
* What are the inputs and labels?
* How would the business manager evaluate success/performance?
* How would we evaluate the performance of an ML algorithm during development?
* How often should we retrain our model?
* Existing resources/libraries/services that we can leverage?
* Next steps?


## Featurization or Feature Extraction
> Mapping raw input `x` to R<sup>d</sup>.

## Loss Function
> A loss function scores how far off a prediction is from the desired "target" output.

### Classification Loss or `0/1` Loss
* Loss is 1 if prediction is wrong, else 0.

### Square Loss for Regression

## Main Principle of Train/Test Split
* Train/Test setup should represent Train/Deploy scenario as closely as possible
  * Random split of labeled data into train/test is usually the right approach
  * Time seriese data: split data in time, rather than randomly

## What Might go Wrong with ML? 
### Nonstationarity
> Nonstationarity: when the thing you are modeling changes over time.
* Nonstationarity Takes Two Forms:
  * **Covariate Shift**: input distribution changed bewteen training and deployment. (Covariate is another term for input feature)
    * e.g. once popular search queries become less popular
  * **Concept Drift**: correct output for given input changes over time.
    * e.g. season changes and given person no longer is interested in winter coats.

### Leakage
> Information about labels sneak into the features.

### Sample Bias
> Test inputs and deployment inputs have different distributions.

## Model Complexity and Overfitting
> Hyperparameter: It is a parameter of the Machine Learning algorithm (which finds the best parameters for a model) itself.
> Overfitting: training performance is good but test/validation performance is poor.

* Fix overfitting
  * Reduce model complexity
  * Get more training data



# Cross-Validation

## What's the Purpose of Cross-Validation?
* (ESL) Cross-Validation is used for estimating *prediction error*.
  * It directly estimates the *expected extra-sample error*: ![expected extra-sample error](resources/ExpectedExtraSampleErrorEqn.gif), the *average generalization error* when the method, ![hat_f(x)](resources/hatfX.gif), is applied to an independent test sample from the join distribution of `X` and `Y`.
  * We might hope that cross-validation estimates the *conditional error*, with the training set ![Tau](resources/UpperTau.gif) held fixed. But cross-validation typically estimates well only the *expected prediction error* (or *average error* E<sub>rr</sub>.

* Cross-Validation can also be used to find the optimal hyperparameter.
  * Given a set of models `f(x,α)` indexed by a tuning parameter `α`, denote by ![The Alphath model fit with the k-th part of the data removed](resources/alphath_model_fit_wo_kth_data.gif) the `αth` model fit with the `kth` part of the data removed. Then for this set of models we define
  
  ![Cross-Validation-with-Alpha-Equation](resources/cv_alpha_eqn.gif)
  
  * The function ![cross_validation_prediction_error](resources/cv_alpha.gif) provides an estimate of the test error curve, and we find the hyperparameter that minimizes it.
    * Often a "one-standard error" rule is used with cross-validation, in which we choose the most parsimonious model whose error is no more than one standard error above the error of the best model.
  * Our final chosen model is ![chosen_model](resources/f_alpha.gif), which we then fit to all the data.

## Types of Cross-Validation
* K-Fold Cross-Validation  
* Leave-One-Out Cross-Validation
  * Special case of K-Fold cross-validation when `K=N`.

## How to Choose K?
* With `K=N`, the cross-validation estimator is approximately unbiased for the trun (expected) prediction error, but have high variance because the N "training sets" are so similar to one another.
  * Leave-one-out cross-validation has low bias but can have high variance. 
* With K smaller, cross-validation estimator has lower variance, but bias could be a problem, depending on how the performance of the learning method varies with the size of the training set. 
* It is thus important to report the estimated standard error of the CV estimate. 

## The Right Way to Do Cross-validation
* In general, with a multistep modeling procedure, cross-validation must be applied to the entire sequence of modeling steps. In particular, samples must be "left out" before any selection or filtering steps are applied. There is one qualification: initial *unsupervised* screening steps (no class labels are involved) can be done before samples are left out. 
  * Leaving samples out (in cross-validation) **after** the variables (features) have been selected (based on all of the samples) does not correctly mimic the application of the classifier to a completely independent test set, since these predictors "have already seen" the left out samples. 

## Other Related
* Forward Chaining is the cross validation for Time Series
* Finite Class Lemma???
  * As you test more and more on a data set, you are more likely to overfit.
  * The size of confidence interval grows with the number of things you are testing proportionately.  

## TODO
* Think about both amount of data and computation needed to learn.
* SVM “is just” ERM with hinge loss with `2 regularization
* Pegasos “is just” SVM with SGD with a particular step size rule
* Random forest “is just” bagging with trees, with an interesting tweak on choosing splitting variables
* Matrix Calclus
* 

* Print out concept checks and problems
