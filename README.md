This repositary holds my solutions to the exercises and problems in book "[Learning from Data: A Short Course](http://work.caltech.edu/telecourse.html)" by Yaser Abu-Mostafa et al.

## Chapter 1: The Learning Problem
#### Exercises
#### Problems

## Chapter 2: Training versus Testing
#### Exercises
#### Problems

## Chapter 3: The Linear Model
#### Exercises
#### Problems

## Chapter 4: Overfitting
#### Exercises
#### Problems

## Chapter 5: Three Learning Principles
#### Exercises
#### Problems

## Chapter 6: Similarity-Based Methods
#### Exercises
#### Problems

## Chapter 7: Neural Networks
#### Exercises
#### Problems

## Chapter 8:  Support Vector Machine
#### Exercises
#### Problems

## Chapter 9: Learning Aides
#### Exercises
#### Problems

## Appendix B: Linear Algebra

## Appendix C: The E-M Algorithm

# Questions
## Chapter 1
* What are the components of a learning algorithm? 
* What are different types of learning problems? 
* Why we are able to learn at all? 
* What are the meanings of various probability quantities? 

## Chapter 2
* How do we generalize from training data? 
* How to understand VC dimension? What is the VC dimension of linear models, e.g. perceptron? What do we use it for?
* How to understand the bounds? 
* How to understand the two competing forces: approximation and generalization?
* How to understand the bias variance trade off? How to derive it? 
* Why do we need test set? What's the difference between train and test set? What are the advantages of using a test set?
* What is learning curve? How to intepret it? What are the typical learning curves for linear regression? 

## Chapter 3
* What are the linear models? Linear classification, linear regression and logistic regression.
* What's the application of approximation-generalization in linear models? 
* Why minimize perceptron requires combinatorial efforts while minimize linear regression requires just analytic solution. Logistic regression needs gradient descent.
* When can GD be used? What algorithms use gradient descent method? What use sub-gradient methods? What are the requirements for GD? 
What are the advantages of using SGD? What does SGD work at all? What's the convergence speed between GD and SGD? 
* Why fixed rate learning rate GD works? 
* How does feature transformation affect the VC dimension? 
* What are the advantages and disadvantages of feature transformation? 
* What is a projection matrix? Give an example of projection in 1-D or 2-D. Where does it project to?


## Chapter 4
* What is overfitting? What causes overfitting? When does it happen? How do we measure it? Why it's important? What can we do to reduce it? How is overfitting related with in-sample error, out-of-sample error etc. ? How does model complexity, number of data points, and hypothesis set affect overfitting? 
* What are the stochastic noise and deterministic noise? What're the differences betwen them? How do they show up in learning algorithm? 
* What is regularization?  How does regularization affect the VC bound? What are the impacts of regularization on generalization error? What are different ways to do regularization? What are the components of regularization? Regularizer, weight decay, etc. 
* What are the Lagrange optimization? Understand it intuitively?
* What is validation? What do we use it for? What's the relationship between validation and regularization? How are they related to the generalization error?
* What is validation set? Why do we need re-train with all training data after we run through the validation set? Why do we need validation set? 
* What is cross-validation? When do we use it? What's the procedure to apply cross-validation?

## Chapter 5
* What is the Occam's razor? 
* How do we measure complexity? What are the two recurring theme in complexity measurement in objects? 
* What is the axim of nonfalsifibility? 
* What is sampling bias? What's the implication for VC bounds and Hoeffding bound? What are other biases we see in learning? 
* What is data snooping? How to deal with data snooping? 




  
