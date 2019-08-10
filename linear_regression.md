* training a linear classifier boils down to finding the best linear combination of features, which are column vectors of the data matrix. The solution space is characterized by the column space and the null space of the data matrix.
The quality of the trained linear classifier directly depends upon the null space and the column space of the data matrix. A large column space means that there is little linear dependency between the features, which is generally good. The null space contains “novel” data points that cannot be formulated as linear combinations of existing data; a large null space could be problematic. 

## Column Space
* Data interpretation:
All outcomes that are linearly predictable based on observed features. The vector w contains the weight of each feature.
* Basis:
The left singular vectors corresponding to nonzero singular values (a subset of the columns of U).
## Row Space
* Data interpretation:
A vector in the row space is something that can be represented as a linear combination of existing data points. Hence, this can be interpreted as the space of “non-novel” data. The vector u contains the weight of each data point in the linear combination.
Basis:
* The right singular vectors corresponding to nonzero singular values (a subset of the columns of V).
## NULL SPACE
* Mathematical definition:
The set of input vectors w where  Aw = 0.
* Mathematical interpretation:
Vectors that are orthogonal to all rows of A. The null space gets squashed to zero by the matrix. This is the “fluff” that adds volume to the solution space of Aw = y.
* Data interpretation:
“Novel” data points that cannot be represented as any linear combination of existing data points.
* Basis:
The right singular vectors corresponding to the zero singular values (the rest of the columns of V).
## LEFT NULL SPACE
* Mathematical definition:
The set of input vectors u where uTA = 0.
* Mathematical interpretation:
Vectors that are orthogonal to all columns of A. The left null space is orthogonal to the column space.
* Data interpretation:
“Novel feature vectors" that are not representable by linear combinations of existing features.
* Basis:
The left singular vectors corresponding to the zero singular values (the rest of the columns of U).

## Others
Column space and row space contain what is already representable based on observed data and features. Those vectors that lie in the column space are non-novel features. Those vectors that lie in the row space are non-novel data points.
For the purposes of modeling and prediction, non-novelty is good. A full column space means that the feature set contains enough information to model any target vector we wish. A full row space means that the different data points contain enough variation to cover all possible corners of the feature space. It’s the novel data points and features—respectively contained in the null space and the left null space—that we have to worry about.

In the application of building linear models of data, the null space can also be viewed as the subspace of “novel” data points. Novelty is not a good thing in this context. Novel data points indicate phantom data that is not linearly representable by the training set. Similarly, the left null space contains novel features that are not representable as linear combinations of existing features.

The null space is orthogonal to the row space. It’s easy to see why. The definition of null space states that w has an inner product of 0 with every row vector in A. Therefore, w is orthogonal to the space spanned by these row vectors, i.e., the row space. Similarly, the left null space is orthogonal to the column space.

## Solving Linear system
In order to train a linear model, loosely speaking we have to find the input weight vector w that maps to the observed output targets y in the system Aw = y, where A is the data matrix.
A singular value of zero in A squashes whatever input was given; there’s no way to retrace its steps and come up with the original input.
In fact, any input that gets squashed to zero could be added to a particular solution and give us another solution. The general solution looks like this:

wgeneral = wparticular + whomogeneous

wparticular is an exact solution to the equation Aw = y. There may or may not be such a solution. If there isn’t, then the system can only be approximately solved. If there is, then y belongs to what’s known as the column space of A. The column space is the set of vectors that A can map to, by taking linear combinations of its columns.

whomogeneous is a solution to the equation Aw = 0.  The set of all whomogeneous vectors forms the null space of A. This is the span of the right singular vectors with singular value 0.

If the null space contains any vectors other than the all-zero vector, then there are infinitely many solutions to the equation Aw = y. 
But if there are many possible answers, then there are many sets of features that are useful for the classification task. It becomes difficult to understand which ones are truly important.

One way to fix the problem of a large null space is to regulate the model by adding additional constraints:

Aw = y,

where w is such that wTw = c.

In general, feature selection methods deal with selecting the most useful features to reduce computation burden, decrease the amount of confusion for the model, and make the learned model more unique. This is the focus of “Feature Selection”.

## unevenness
When we train a linear classifier, we care not only that there is a general solution to the linear system, but also that we can find it easily. Typically, the training process employs a solver that works by calculating a gradient of the loss function and walking downhill in small steps. When some singular values are very large and others very close to zero, the solver needs to carefully step around the longer singular vectors (those that correspond to large singular values) and spend a lot of time digging around in the shorter singular vectors to find the true answer. This “unevenness” in the spectrum is measured by the condition number of the matrix, which is basically the ratio between the largest and the smallest absolute value of the singular values.

To summarize, in order for there to be a good linear model that is relatively unique, and in order for it to be easy to find, we wish for the following:

1. The label vector can be well approximated by a linear combination of a subset of features (column vectors). Better yet, the set of features should be linearly independent.
1. In order for the null space to be small, the row space must be large. (This is due to the fact that the two subspaces are orthogonal.) The more linearly independent the set of data points (row vectors), the smaller the null space.
1. In order for the solution to be easy to find, the condition number of the data matrix—the ratio between the maximum and minimum singular values—should be small.
