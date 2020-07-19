This repositary holds my solutions to the exercises and problems in book "[Learning from Data: A Short Course](http://work.caltech.edu/telecourse.html)" by Yaser Abu-Mostafa et al.

## Chapter 1: [The Learning Problem](Solutions%20to%20Chapter%201%20The%20Learning%20Problem.ipynb)

Missing: Problem 1.7 (b) 

## Chapter 2: [Training versus Testing](Solutions%20to%20Chapter%202%20Training%20versus%20Testing.ipynb)

Missing: 
* Exercises: 2.4
* Problems: 2.4, 2.9, 2.10, 2.14, 2.15, 2.19

## Chapter 3: [The Linear Model](Solutions%20to%20Chapter%203%20The%20Linear%20Model.ipynb)

Missing:
* Exercises: 3.12, 3.15
* Problems: 3.15 (c), 3.18

## Chapter 4: [Overfitting](Solutions%20to%20Chapter%204%20Overfitting.ipynb)

Missing:
* Exercises: 4.9, 4.10
* Problems: 4.4 (f), 4.21

## Chapter 5: [Three Learning Principles](Solutions%20to%20Chapter%205%20Three%20Learning%20Principles.ipynb)


## Chapter 6: [Similarity-Based Methods](Solutions%20to%20Chapter%206%20Similarity-Based%20Methods.ipynb)

Missing:
* Exercises: 6.3
* Problems: 6.5, 6.9, 6.11 (b), 6.15

## Chapter 7: [Neural Networks](Solutions%20to%20Chapter%207%20Neural%20Networks.ipynb)

Missing:
* Exercises: 
* Problems: 7.2, 7.5, 7.9, 7.12, 7.16

## Chapter 8:  [Support Vector Machine](Solutions%20to%20Chapter%208%20Support%20Vector%20Machine.ipynb)

Missing:
* Exercises: 
* Problems: 8.9 (c) 

## Chapter 9: [Learning Aides](Solutions%20to%20Chapter%209%20Learning%20Aides.ipynb)
Missing:
* Exercises: 9.17
* Problems: 9.11, 9.16, 9.17, 9.26, 9.27, 9.28

## Appendix B: [Linear Algebra](Appendix%20B%20Linear%20Algebra.ipynb)

## Appendix C: [The E-M Algorithm](Appendix%20C%20The%20E-M%20Algorithm.ipynb)

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




  
