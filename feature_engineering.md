* Deal with **class-imbalanced dataset**. Imbalanced datasets are problematic for modeling because the model will expend most of its effort fitting to the larger class. Since we have plenty of data in both classes, a good way to resolve the problem is to downsample the larger class (restaurants) to be roughly the same size as the smaller class (nightlife).
* It’s essential to tune hyperparameters when comparing models or features. The default settings of a software package will always return a model.
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




# Evaluation of Feature Engineering Procedure
One needs to have a metric of machine learning model performance to evaluate the effectiveness of a feature engineering procedure. 
First obtain a baseline performance, and compare performance against it after the feature engineering procedure.

# Understand Data

## The four levels of data
### The nominal level

It has the weakest structure. It is discrete and order-less. It consists of data that are purely described by name. Basic examples include blood type (A, O, AB), species of animal, or names of people. These types of data are all qualitative.
1. Count the number of different values
```
df.value_counts().index[0]
```
2. Plot the bar chart ('bar') or pie chart ('pie')
```
df['col_name'].value_counts().sort_values(ascending=False).head(20).plot(kind='bar')
```

### The ordinal level

The ordinal scale inherits all of the properties of the nominal level, but has important additional properties:
Data at the ordinal level can be naturally ordered
This implies that some data values in the column can be considered better than or greater than others

As with the nominal level, data at the ordinal level is still categorical in nature, even if numbers are used to represent the categories.
1. Median and percentiles
2. Stem-and-leaf plots
3. Box plot
```
df['col'].value_counts().plot(kind='box')
```

### The interval level

At the interval data level, we are working with numerical data that not only has ordering like at the ordinal level, but also has meaningful differences between values. This means that at the interval level, not only may we order and compare values, we may also add and subtract values.
1. Check number of unique values
```
df['col'].nunique()
```
2. Plot histogram, use sharex=True to put all x-axis in one scale
```
df['col'].hist(by=df['val'], sharex=True, sharey=True, figsize=(10, 10), bins=20)
```
3. Plot mean values
```
df.groupby('val')['col'].mean().plot(kind='line')
```
4. Scatter plot of two columns
```
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(df['col1'], df['col2'])
plt.show()
```
5. Plot with groupby
```
df.groupby('col1').mean()['col2'].plot()
```
6. Rolling smoothing
```
f.groupby('col1').mean()['col2'].rolling(20).mean().plot()
```

### The ratio level

Now we have a notion of true zero which gives us the ability to multiply and divide values. It allows ratio statement.
1. Bar chart
```
fig = plt.figure(figsize=(15,5))
ax = fig.gca()

df.groupby('col1')[['col2']].mean().sort_values('col2', ascending=False).tail(20).plot.bar(stacked=False, ax=ax, color='darkorange')
```
## Numerical Data
1. The first sanity check for numeric data is whether the magnitude matters. Do we just need to know whether it’s positive or negative? Or perhaps we only need to know the magnitude at a very coarse granularity? This sanity check is particularly important for automatically accrued numbers such as counts—the number of daily visits to a website, the number of reviews garnered by a restaurant, etc.
1. Next, consider the scale of the features. Models that are smooth functions of input features are sensitive to the scale of the input.
Logical functions, on the other hand, are not sensitive to input feature scale. Another example of a logical function is the step function (e.g., is input x greater than 5?). Decision tree models consist of step functions of input features. Hence, models based on space-partitioning trees (decision trees, gradient boosted machines, random forests) are not sensitive to scale. 

The only exception is if the scale of the input grows over time, which is the case if the feature is an accumulated count of some sort—eventually it will grow outside of the range that the tree was trained on. If this might be the case, then it might be necessary to rescale the inputs periodically. Another solution is the bin-counting method

1. It’s also important to consider the distribution of numeric features. 
The distribution of input features matters to some models more than others. For instance, the training process of a linear regression model assumes that prediction errors are distributed like a Gaussian. This is usually fine, except when the prediction target spreads out over several orders of magnitude. In this case, the Gaussian error assumption likely no longer holds. One way to deal with this is to transform the output target in order to tame the magnitude of the growth. (Strictly speaking this would be target engineering, not feature engineering.) Log transforms, which are a type of power transform, take the distribution of the variable closer to Gaussian.

### Feature space vs Data space
Collectively, a collection of data can be visualized in feature space as a point cloud. Conversely, we can visualize features in data space. 
* Picture the set of data points in feature space. Each data point is a dot, and the whole set of data points forms a blob.

### Dealing with Counts
It is a good idea to check the scale and determine whether to keep the data as raw numbers, convert them into binary values to indicate presence, or bin them into coarser granularity. 

1. Binarization
1. Quantization or Binning. Raw counts that span several orders of magnitude are problematic for many models. In a linear model, the same linear coefficient would have to work for all possible values of the count. Large counts could also wreak havoc in unsupervised learning methods such as k-means clustering, which uses Euclidean distance as a similarity function to measure the similarity between data points. A large count in one element of the data vector would outweigh the similarity in all other elements, which could throw off the entire similarity measurement.
One solution is to contain the scale by quantizing the count. In other words, we group the counts into bins, and get rid of the actual count values. Quantization maps a continuous number to a discrete one. We can think of the discretized numbers as an ordered sequence of bins that represent a measure of intensity.

In order to quantize data, we have to decide how wide each bin should be. The solutions fall into two categories: fixed-width or adaptive. 
   1. FIXED-WIDTH BINNING: With fixed-width binning, each bin contains a specific numeric range. The ranges can be custom designed or automatically segmented, and they can be linearly scaled or exponentially scaled. To map from the count to the bin, we simply divide by the width of the bin and take the integer part. When the numbers span multiple magnitudes, it may be better to group by powers of 10 (or powers of any constant): 0–9, 10–99, 100–999, 1000–9999, etc. The bin widths grow exponentially, going from O(10), to O(100), O(1000), and beyond. To map from the count to the bin, we take the log of the count. Exponential-width binning is very much related to the log transform
   1. QUANTILE BINNING: But if there are large gaps in the counts, then there will be many empty bins with no data. This problem can be solved by adaptively positioning the bins based on the distribution of the data. This can be done using the quantiles of the distribution. 
  
### Log Transformation
The log function compresses the range of large numbers and expands the range of small numbers. The larger x is, the slower log(x) increments. The log transform is a powerful tool for dealing with positive numbers with a heavy-tailed distribution.

### Power Transforms: Generalization of the Log Transform
The log transform is a specific example of a family of transformations known as power transforms. In statistical terms, these are variance-stabilizing transformations. To understand why variance stabilization is good, consider the Poisson distribution. This is a heavy-tailed distribution with a variance that is equal to its mean: hence, the larger its center of mass, the larger its variance, and the heavier the tail. Power transforms change the distribution of the variable so that the variance is no longer dependent on the mean. For example, suppose a random variable X has the Poisson distribution. If we transform X by taking its square root, the variance of X˜=sqrt(X) is roughly constant, instead of being equal to the mean.

A simple generalization of both the square root transform and the log transform is known as the Box-Cox transform
Setting λ to be less than 1 compresses the higher values, and setting λ higher than 1 has the opposite effect.

The Box-Cox formulation only works when the data is positive. For nonpositive data, one could shift the values by adding a fixed constant. When applying the Box-Cox transformation or a more general power transform, we have to determine a value for the parameter λ. This may be done via maximum likelihood (finding the λ that maximizes the Gaussian likelihood of the resulting transformed signal) or Bayesian methods. A full treatment of the usage of Box-Cox and general power transforms is outside the scope of this book. Interested readers may find more information on power transforms in Econometric Methods by Johnston and DiNardo (1997).

* A probability plot, or probplot, is an easy way to visually compare an empirical distribution of data against a theoretical distribution. This is essentially a scatter plot of observed versus theoretical quantiles. 

## Feature Scaling or Feature Normalization
* No matter what the scaling method, feature scaling always divides the feature by a constant (known as the normalization constant). Therefore, it does not change the shape of the single-feature distribution. 
* DON’T “CENTER” SPARSE DATA!
   Use caution when performing min-max scaling and standardization on sparse features. Both subtract a quantity from the original feature value. For min-max scaling, the shift is the minimum over all values of the current feature; for standardization, it is the mean. If the shift is not zero, then these two transforms can turn a sparse feature vector where most values are zero into a dense one. This in turn could create a huge computational burden for the classifier, depending on how it is implemented (not to mention that it would be horrendous if the representation now included every word that didn’t appear in a document!). Bag-of-words is a sparse representation, and most classification libraries optimize for sparse inputs.

### Min-Max Scaling
### Standardization (Variance Scaling)
### ℓ^2 Normalization or ℓ2 scaling
* This technique normalizes (divides) the original feature value by the ℓ^2 norm or the Euclidean norm. The ℓ2 norm sums the squares of the values of the features across data points, then takes the square root. After ℓ2 normalization, the feature column has norm 1. 

## Interaction Features
* A simple pairwise interaction feature is the product of two features. The analogy is the logical AND.
Decision tree–based models get this for free, but generalized linear models often find interaction features very helpful.

* The training and scoring time of a linear model with pairwise interaction features would go from O(n) to O(n2), where n is the number of singleton features.

There are a few ways around the computational expense of higher-order interaction features. One could perform feature selection on top of all of the interaction features. Alternatively, one could more carefully craft a smaller number of complex features.

## Feature Selection
### Filtering

Filtering techniques preprocess features to remove ones that are unlikely to be useful for the model. For example, one could compute the correlation or mutual information between each feature and the response variable, and filter out the features that fall below a threshold.

Filtering techniques are much cheaper than the wrapper techniques described next, but they do not take into account the model being employed. Hence, they may not be able to select the right features for the model. It is best to do prefiltering conservatively, so as not to inadvertently eliminate useful features before they even make it to the model training step.

### Wrapper methods

The wrapper method treats the model as a black box that provides a quality score of a proposed subset for features. There is a separate method that iteratively refines the subset.

### Embedded methods
These methods perform feature selection as part of the model training process. For example, a decision tree inherently performs feature selection because it selects one feature on which to split the tree at each training step. Another example is the ℓ1 regularizer, which can be added to the training objective of any linear model. The ℓ1 regularizer encourages models that use a few features as opposed to a lot of features, so it’s also known as a sparsity constraint on the model. Embedded methods incorporate feature selection as part of the model training process. 

A full treatment of feature selection is outside the scope of this book. Interested readers may refer to the survey paper by Guyon and Elisseeff (2003).
















## Structured Data Type
### Basic Checks

1. Count the number of rows, missing values, data types etc
```df.info()```
2. Count how many missing values in each column
```df.isnull().sum()```
3. Check descriptive statistics on quantitative columns
```df.describe()```
4. Check the differences of histograms for data in category 1 vs. category 2.
5. Check correlation map
```
# look at the heatmap of the correlation matrix of data
import seaborn as sns
sns.heatmap(df.corr())
```



## Data Cleanup
### Remove rows with missing values in them
Most time this strategy is not very good.
### Impute (fill in) missing values
Use pipeline to fill with mean or median etc.
```
from sklearn.preprocessing import Imputer
```

### Data normalization
1. Z-score standardization
1. Min-max scaling
1. Row normalization: It ensure that each row of data has a unit norm, meaning that each row will be the same vector length.

### A list of some popular learning algorithms that are affected by the scale of data
1. KNN-due to its reliance on the Euclidean Distance
1. K-Means Clustering - same reasoning as KNN
1. Logistic regression, SVM, neural networks - if you are using gradient descent to learn weights
1. Principal component analysis - eigen vectors will be skewed towards larger columns
1. RBF Kernels, and anything that uses the Euclidean distance.

### Encoding Categorical Data
1. Encoding at the nominal level
   1. Transform our categorical data into dummy variables. 
The dummy variable trap is when you have independent variables that are multicollinear, or highly correlated. Simply put, these variables can be predicted from each other. So, in our gender example, the dummy variable trap would be if we include both female as (0|1) and male as (0|1), essentially creating a duplicate category. It can be inferred that a 0 female value indicates a male.

1. Encoding at the ordinal level
To maintain the order, we will use a label encoder. 

### Bucketing Continuous features into categories
```
pandas.cut
```

# Feature Construction
## Polynomial Features
A key method of working with numerical data and creating more features is through scikit-learn's PolynomialFeatures class. In its simplest form, this constructor will create new columns that are products of existing columns to capture feature interactions.

## Text Specific Feature Construction
1. Bag of words representation
   1. Tokenizing
   1. Counting
   1. Normalizing

1. CountVectorizer
   It converts text columns into matrices where columns are tokens and cell values are counts of occurrences of each token in each document. The resulting matrix is referred to as a document-term matrix because each row will represent a document (in this case, a tweet) and each column represents a term (a word).
1. The Tf-idf vectorizer

# Feature Selection
* If your features are mostly categorical, you should start by trying to implement a SelectKBest with a Chi2 ranker or a tree-based model selector.
* If your features are largely quantitative, using linear models as model-based selectors and relying on correlations tends to yield greater results.
* If you are solving a binary classification problem, using a Support Vector Classification model along with a SelectFromModel selector will probably fit nicely, as the SVC tries to find coefficients to optimize for binary classification tasks.
* A little bit of EDA can go a long way in manual feature selection. The importance of having domain knowledge in the domain from which the data originated cannot be understated.

## Statistical-based
Model-based selection relies on a preprocessing step that involves training a secondary machine learning model and using that model's predictive power to select features.
### Pearson correlation

We will assume that the more correlated a feature is to the response, the more useful it will be. Any feature that is not as strongly correlated will not be as useful to us.
It is worth noting that Pearson's correlation generally requires that each column be normally distributed (which we are not assuming). We can also largely ignore this requirement because our dataset is large (over 500 is the threshold).
Correlation coefficients are also used to determine feature interactions and redundancies. A key method of reducing overfitting in machine learning is spotting and removing these redundancies.
### Hypothesis testing

Feature selection via hypothesis testing will attempt to select only the best features from a dataset, these tests rely more on formalized statistical methods and are interpreted through what are known as p-values. 
In the case of feature selection, the hypothesis we wish to test is along the lines of: True or False: This feature has no relevance to the response variable. We want to test this hypothesis for every feature and decide whether the features hold some significance in the prediction of the response. 
Simply put, the lower the p-value, the better the chance that we can reject the null hypothesis. For our purposes, the smaller the p-value, the better the chances that the feature has some relevance to our response variable and we should keep it.
```
# SelectKBest selects features according to the k highest scores of a given scoring function
from sklearn.feature_selection import SelectKBest

# This models a statistical test known as ANOVA
from sklearn.feature_selection import f_classif

# f_classif allows for negative values, not all do
# chi2 is a very common classification criteria but only allows for positive values
# regression has its own statistical tests
```
The big take away from this is that the f_classif function will perform an ANOVA test (a type of hypothesis test) on each feature on its own (hence the name univariate testing) and assign that feature a p-value. The SelectKBestwill rank the features by that p-value (the lower the better) and keep only the best k (a human input) features. Let's try this out in Python.

## Model-based
1. Tree-based model feature importance
1. Linear model's coef_ attribute

# Feature Transformation
It's a suite of algorithms designed to alter the internal structure of data to produce mathematically superior super-columns. The toughest part of feature transformations is the suspension of our belief that the original feature space is the best. We must be open to the fact that there may be other mathematical axes and systems that describe our data just as well with fewer features, or possibly even better. Feature transformation algorithms are able to construct new features by selecting the best of all columns and combining this latent structure with a few brand new columns
## Dimension Deduction
### PCA
1. A scree plot is a simple line graph that shows the percentage of total variance explained in the data by each principal component. To build this plot, we will sort the eigenvalues in order of descending value and plot the cumulative variance explained by each component and all components prior.
1. Centering data doesn't affect the principal components. The reason this is happening is because matrices have the same covariance matrix as their centered counterparts. If two matrices have the same covariance matrix, then they will have the same eignenvalue decomposition.
1. PCA is scale-invariant, meaning that scale affects the components. Note that when we say scaling, we mean centering and dividing by the standard deviation. It's because once we scaled our data, the columns' covariance with one another became more consistent and the variance explained by each principal component was spread out instead of being solidified in a single PC. In practice and production, we generally recommend scaling, but it is a good idea to test your pipeline's performance on both scaled and un-scaled data.
1. PCA 1, our first principal component, should be carrying the majority of the variance within it, which is why the projected data is spread out mostly across the new x axis
1. The assumption that we were making was that the original data took on a shape that could be decomposed and represented by a single linear transformation (the matrix operation).

### Linear Discriminant Analysis

1. The main difference between LDA and PCA is that instead of focusing on the variance of the data as a whole like PCA, LDA optimizes the lower-dimensional space for the best class separability. This means that the new coordinate system is more useful in finding decision boundaries for classification models, which is perfect for us when building classification pipelines.
1. The reason that LDA is extremely useful is that separating based on class separability helps us avoid overfitting in our machine learning pipelines. This is also known as preventing the curse of dimensionality. LDA also reduces computational costs.
1. The way LDA is trying to work is by drawing decision boundaries between our classes. Because we only have three classes in the iris, we may only draw up to two decision boundaries. In general, fitting LDA to a dataset with n classes will only produce up to n-1 components in eigenvalues, i.e. the rest eigen values are close to zero.
1. This is because the goal of scalings_ is not to create a new coordinate system, but just to point in the direction of boundaries in the data that optimizes for class separability.
1. It is sufficient to understand that the main difference between PCA and LDA is that PCA is an unsupervised method that captures the variance of the data as a whole whereas LDA, a supervised method, uses the response variable to capture class separability.
1. It is common to correctly use all three of these algorithms in the same pipelines and perform hyper-parameter tuning to fine-tune the process. This shows us that more often than not, the best production-ready machine learning pipelines are in fact a combination of multiple feature engineering methods.

### PCA and LDA are extremely powerful tools, but have limitations.
1. Both of them are linear transformations, which means that they can only create linear boundaries and capture linear qualities in our data. 
1. They are also static transformations. No matter what data we input into a PCA or LDA, the output is expected and mathematical. If the data we are using isn't a good fit for PCA or LDA (they exhibit non-linear qualities, for example, they are circular), then the two algorithms will not help us, no matter how much we grid search.

# Feature Learning
It focuses on feature learning using non-parametric algorithms (those that do not depend on the shape of the data) to automatically learn new features. They do not make any assumptions about the shape of the incoming data and rely on stochastic learning.
instead of throwing the same equation at the matrix of data every time, they will attempt to figure out the best features to extract by looking at the data points over and over again (in epochs) and converge onto a solution (potentially different ones at runtime).
## No-parametric fallacy
It is important to mention that a model being non-parametric doesn't mean that there are no assumptions at all made by the model during training.
While the algorithms that we will be introducing in this chapter forgo the assumption on the shape of the data, they still may make assumptions on other aspects of the data, for example, the values of the cells.
They all involve learning brand new features from raw data. They then use these new features to enhance the way that they interact with data.

## Restricted Boltzmann Machine
1. A simple deep learning architecture that is set up to learn a set number of new dimensions based on a probabilistic model that data follows.
1. The features that are extracted by RBMs tend to work best when followed by linear models such as linear regression, logistic regression, perceptron's, and so on.
1. The restriction in the RBM is that we do not allow for any intra-layer communication. This lets nodes independently create weights and biases that end up being (hopefully) independent features for our data.

## Word Embedding

Mastering a subject is not just about knowing the definitions and being able to derive the formulas. It is not enough to know how the mechanism works and what it can do - one must also understand why it is designed that way, how it relates to other techniques, and what the pros and cons of each approach are. Mastery is about knowing precisely how something is done, having an intuition for the underlying principles, and integrating it into one's existing web of knowledge. One does not become a master of something by simply reading a book, though a good book can open new doors. It has to involve practice - putting the ideas to use, which is an iterative process. With every iteration, we know the ideas better and become increasingly more adept and creative at applying them. The goal of this book is to facilitate the application of its ideas.


# Text Data: Flattening, Filtering, and Chunking

## Bag of Words
* The ordering of words in the vector is not important, as long as it is consistent for all documents in the dataset. Neither does bag-of-words represent any concept of word hierarchy.

* Sometimes it is also informative to look at feature vectors in data space. A feature vector contains the value of the feature in each data point. The axes denote individual data points, and the points denote feature vectors.
With bag-of-words featurization for text documents, a feature is a word, and a feature vector contains the counts of this word in each document. 
In this way, a word is represented as a “bag-of-documents.”  As we shall see in Chapter 4, these bag-of-documents vectors come from the matrix transpose of the bag-of-words vectors.

* Bag-of-words is not perfect. Breaking down a sentence into single words can destroy the semantic meaning.

## Bag-of-n-Grams
* An n-gram is a sequence of n tokens. A word is essentially a 1-gram, also known as a unigram. After tokenization, the counting mechanism can collate individual tokens into word counts, or count overlapping sequences as n-grams.

* n-grams retain more of the original sequence structure of the text, and therefore the bag-of-n-grams representation can be more informative. However, this comes at a cost. Theoretically, with k unique words, there could be k2 unique 2-grams (also called bigrams). 

* Bag-of-n-grams generates a lot more distinct n-grams. It increases the feature storage cost, as well as the computation cost of the model training and prediction stages. The number of data points remains the same, but the dimension of the feature space is now much larger. Hence, the data is much more sparse. The higher n is, the higher the storage and computation cost, and the sparser the data. For these reasons, longer n-grams do not always lead to improvements in model accuracy (or any other performance measure). People usually stop at n = 2 or 3. Longer n-grams are rarely used.


## Filtering for Cleaner Features

### Stopwords
Stopword lists are a way of weeding out common words that make for vacuous features.

### Frequency-Based Filtering

1. FREQUENT WORDS
Frequency statistics are great for filtering out corpus-specific common words as well as general-purpose stopwords.

1. RARE WORDS
Rare words incur a large computation and storage cost for not much additional gain.Rare words can be easily identified and trimmed based on word count statistics. Alternatively, their counts can be aggregated into a special garbage bin, which can serve as an additional feature.

1. Stemming
Stemming is an NLP task that tries to chop each word down to its basic linguistic word stem form.

## Atoms of Meaning: From Words to n-Grams to Phrases
One way to combat the increase in sparsity and cost in bag-of-n-grams is to filter the n-grams and retain only the most meaningful phrases. 

### Parsing and Tokenization
1. Parsing is necessary when the string contains more than plain text.
1. Tokenization turns the string—a sequence of characters—into a sequence of tokens. Each token can then be counted as a word. 

### Collocation Extraction for Phrase Detection

* In computational natural language processing (NLP), the concept of a useful phrase is called a collocation. In the words of Manning and Schütze (1999: 151), “A collocation is an expression consisting of two or more words that correspond to some conventional way of saying things.” 

* Collocations are more meaningful than the sum of their parts.  Not every collocation is an n-gram. Conversely, not every n-gram is deemed a meaningful collocation.

#### How to discover and extract collocations from text? 
1. FREQUENCY-BASED METHODS
A simple hack is to look at the most frequently occurring n-grams. 

1. HYPOTHESIS TESTING FOR COLLOCATION EXTRACTION
The key idea is to ask whether two words appear together more often than they would by chance. Hypothesis testing is a way to boil noisy data down to “yes” or “no” answers. It involves modeling the data as samples drawn from random distributions. 

### CHUNKING AND PART-OF-SPEECH TAGGING
* To generate longer phrases, there are other methods such as chunking or combining with part-of-speech (PoS) tagging.Chunking is a bit more sophisticated than finding n-grams, in that it forms sequences of tokens based on parts of speech, using rule-based models.

## TF-IDF (term frequency–inverse document frequency)
1. tf-idf makes rare words more prominent and effectively ignores common words. It is closely related to the frequency-based filtering methods. Tf-idf transforms word count features through multiplication with a constant. Hence, it is an example of feature scaling.
```
bow(w, d) = # times word w appears in document d
tf-idf(w, d) = bow(w, d) * log(N / (# documents in which word w appears))
N is the total number of documents in the dataset. 
```

## Impacts of Feature Scaling on Linear Models

1. For linear models like logistic regression, the features are used through the "data matrix". The columns represent all possible words in the vocabulary. The rows represent each document. It contains data points represented as fixed-length flat vectors. With bag-of-words vectors, the data matrix is also known as the "document-term matrix".

1. Feature scaling methods are essentially column operations on the data matrix. In particular, tf-idf and ℓ2 normalization both multiply the entire column (an n-gram feature, for example) by a constant.

1. The null space of the data matrix can be large for a couple of reasons. 
   * First, many datasets contain data points that are very similar to one another. This means the effective row space is small compared to the number of data points in the dataset. 
   * Second, the number of features can be much larger than the number of data points.
   * Moreover, the number of distinct words usually grows with the number of documents in the dataset, so adding more documents would not necessarily decrease the feature-to-data ratio or reduce the null space.

1. With bag-of-words (no feature scaling), the column space is relatively small compared to the number of features. There could be words that appear roughly the same number of times in the same documents. This would lead to the corresponding column vectors being nearly linearly dependent, which leads to the column space being not as full rank as it could be (see Appendix A for the definition of full rank). This is called a **rank deficiency**.

1. Rank-deficient row space and column space lead to the model being overly provisioned for the problem. The linear model outfits a weight parameter for each feature in the dataset. If the row and column spaces were full rank (Strictly speaking, the row space and column space for a rectangular matrix cannot both be full rank. The maximum rank for both subspaces is the smaller of m (the number of rows) and n (the number of columns).), then the model would allow us to generate any target vector in the output space. When they are rank deficient, the model has more degrees of freedom than it needs. This makes it harder to pin down a solution.

### Can feature scaling solve the rank deficiency problem of the data matrix? 
1. The column space is defined as the linear combination of all column vectors (boldface indicates a vector): ```a1v1 + a2v2 + ... + anvn```. Feature scaling replaces a column vector with a constant multiple, say v˜1=cv1. But we can still generate the original linear combination by just replacing a1 with a˜1=a1/c. It appears that feature scaling does not change the rank of the column space. Similarly, feature scaling does not affect the rank of the null space, because one can counteract the scaled feature column by reverse scaling the corresponding entry in the weight vector.

However, as usual, there is one catch. If the scalar is 0, then there is no way to recover the original linear combination; v1 is gone. If that vector is linearly independent from all the other columns, then we’ve effectively shrunk the column space and enlarged the null space.

If that vector is not correlated with the target output, then this is effectively pruning away noisy signals, which is a good thing. This turns out to be the key difference between tf-idf and ℓ2 normalization. ℓ2 normalization would never compute a norm of zero, unless the vector contains all zeros. If the vector is close to zero, then its norm is also close to zero. Dividing by the small norm would accentuate the vector and make it longer.

Tf-idf, on the other hand, can generate scaling factors that are close to zero, as shown in Figure 4-2. This happens when the word is present in a large number of documents in the training set. Such a word is likely not strongly correlated with the target vector. Pruning it away allows the solver to focus on the other directions in the column space and find better solutions (although the improvement in accuracy will probably not be huge, because there are typically few noisy directions that are prunable in this way).

Where feature scaling—both ℓ2 and tf-idf—does have a telling effect is on the convergence speed of the solver. This is a sign that the data matrix now has a much smaller condition number (the ratio between the largest and smallest singular values—see Appendix A for a full discussion of these terms). In fact, ℓ2 normalization makes the condition number nearly 1. But it’s not the case that the better the condition number, the better the solution. During this experiment, ℓ2 normalization converged much faster than either BoW or tf-idf. But it is also more sensitive to overfitting: it requires much more regularization and is more sensitive to the number of iterations during optimization.

Tf-idf and ℓ2 normalization do not improve the final classifier’s accuracy above plain bag-of-words. After acquiring some statistical modeling and linear algebra chops, we realize why: neither of them changes the column space of the data matrix.

One small difference between the two is that tf-idf can “stretch” the word count as well as “compress” it. In other words, it makes some counts bigger, and others close to zero. Therefore, tf-idf could altogether eliminate uninformative words.

Along the way, we also discovered another effect of feature scaling: it improves the condition number of the data matrix, making linear models much faster to train. Both ℓ2 normalization and tf-idf have this effect.

To summarize, the lesson is: the right feature scaling can be helpful for classification. The right scaling accentuates the informative words and downweights the common words. It can also improve the condition number of the data matrix. The right scaling is not necessarily uniform column scaling.

## Categorical Variables
### Encoding Categorical Variables
1. It is tempting to simply assign an integer, say from 1 to k, to each of k possible categories—but the resulting values would be orderable against each other, which should not be permissible for categories.
1. One-hot Encoding
   1. There will be one feature column for each category for a total of k feature columns.
   1. The problem with one-hot encoding is that it allows for k degrees of freedom, while the variable itself needs only k–1.
   1. One-hot encoding is redundant, which allows for multiple valid models for the same problem. In one-hot encoding “the sum of all bits must be equal to 1”. This introduces a linear dependency here. Linear dependent features are slightly annoying because they mean that the trained linear models will not be unique. Different linear combinations of the features can make the same predictions, so we would need to jump through extra hoops to understand the effect of a feature on the prediction.
   1. With one-hot encoding, the intercept term represents the global mean of the target variable y, and each of the linear coefficients represents how much that category’s average y differs from the global mean.
   1. The advantage is that each feature clearly corresponds to a category. 
   1. Moreover, missing data can be encoded as the all-zeros vector, and the output should be the overall mean of the target variable.
   1. Space requirement:	O(n) using the sparse vector format, where n is the number of data points
   1. Computation requirement: O(nk) under a linear model, where k is the number of categories
   1. Pros
      1. Easiest to implement; Potentially most accurate; Feasible for online learning
   1. Cons
      1. Computationally inefficient; Does not adapt to growing categories; Not feasible for anything other than linear models; Requires large-scale distributed optimization with truly large datasets
    





1. Dummy Encoding
   1. Dummy coding removes the extra degree of freedom by using only k–1 features in the representation. One feature is thrown under the bus and represented by the vector of all zeros. This is known as the **reference category**.
   1. There will be total k-1 feature columns, with zeroes for the records of the reference category in all k-1 feature columns.
   1. With dummy coding, the bias coefficient represents the mean value of the response variable y for the reference category. The coefficient for the *i*th feature is equal to the difference between the mean response value for the *i*th category and the mean of the reference category. It encodes the effect of each category relative to the reference category, which may look strange.
   1. Dummy coding is not redundant. They give rise to unique and interpretable models.
   1. It cannot easily handle missing data, since the all-zeros vector is already mapped to the reference category. 



1. Effect Coding
   1. Effect coding is very similar to dummy coding, with the difference that the reference category is now represented by the vector of all –1’s. 
   1. There will be total k-1 feature columns, with -1's for the records of the reference category in all k-1 feature columns.
   1. The results in linear regression models that are even simpler to interpret. The intercept term represents the global mean of the target variable, and the individual coefficients indicate how much the means of the individual categories differ from the global mean. (This is called the main effect of the category or level, hence the name “effect coding.”) One-hot encoding actually came up with the same intercept and coefficients, but in that case there are linear coefficients for each category. In effect coding, no single feature represents the reference category, so the effect of the reference category needs to be separately computed as the negative sum of the coefficients of all other categories.
   1. Effect coding is not redundant. They give rise to unique and interpretable models.
   1. Effect coding avoids the problem of all-zero vector by using a different code for the reference category, but the vector of all –1’s is a dense vector, which is expensive for both storage and computation.

### Dealing with Large Categorical Variables
1. All three encoding techniques (One-hot, Dummy, Effect encoding) break down when the number of categories becomes very large. Different strategies are needed to handle extremely large categorical variables. The challenge is to find a good feature representation that is memory efficient, yet produces accurate models that are fast to train. 

#### Solutions to large categorical variables
1. Do nothing fancy with the encoding. Use a simple model that is cheap to train. Feed one-hot encoding into a linear model (logistic regression or linear support vector machine) on lots of machines.

1. Compress the features. 
   1. Feature hashing, popular with linear models
      1. A variation of feature hashing adds a sign component, so that counts are either added to or subtracted from the hashed bin. Statistically speaking, this ensures that the inner products between hashed features are equal in expectation to those of the original features.
      1. Feature hashing can be used for models that involve the inner product of feature vectors and coefficients, such as linear models and kernel methods. It has been demonstrated to be successful in the task of spam filtering
      1. One downside to feature hashing is that the hashed features, being aggregates of original features, are no longer interpretable.
      1. Space requirement: O(n) using the sparse matrix format, where n is the number of data points
      1. Computation requirement:	O(nm) under a linear or kernel model, where m is the number of hash bins
      1. Pros: Easy to implement; Makes model training cheaper; Easily adaptable to new categories; Easily handles rare categories; Feasible for online learning
      1. Cons: Only suitable for linear or kernelized models; Hashed features not interpretable; Mixed reports of accuracy





   1. Bin counting, popular with linear models as well as trees
      1. Rather than using the value of the categorical variable as the feature, instead use the conditional probability of the target under that value. In other words, instead of encoding the identity of the categorical value, we compute the association statistics between that value and the target that we wish to predict. For those familiar with naive Bayes classifiers, this statistic should ring a bell, because it is the conditional probability of the class under the assumption that all features are independent. Bin counting assumes that historical data is available for computing the statistics.
      1. We can include other features in addition to the historical probability: the raw counts themselves , the log-odds ratio, or any other derivatives of probability.
      1. The odds ratio is usually defined between two binary variables. It looks at their strength of association by asking the question, “How much more likely is it for Y to be true when X is true?” For instance, we might ask, “How much more likely is Alice to click on an ad than the general population?” Here, X is the binary variable “Alice is the current user,” and Y is the variable “click on ad or not.” The computation uses what’s called the two-way contingency table (basically, four numbers that correspond to the four possible combinations of X and Y). Given an input variable X and a target variable Y, the odds ratio is defined as:
   odds ratio=[P(Y=1|X=1)/P(Y=0|X=1)]/[P(Y=1|X=0)/P(Y=0|X=0)]
   In our example, this translates as the ratio between “how much more likely is it that Alice clicks on an ad rather than does not click” and “how much more likely is it that other people click rather than not click.” 
      1. More simply, we can just look at the numerator, which examines how much more likely it is that a single user (Alice) clicks on an ad versus not clicking. This is suitable for large categorical variables with many values, not just two
      1. In terms of implementation, bin counting requires storing a map between each category and its associated counts. (The rest of the statistics can be derived on the fly from the raw counts.) Hence it requires O(k) space, where k is the number of unique values of the categorical variable.
      1. Space requirement: O(n+k) for small, dense representation of each data point, plus the count statistics that must be kept for each category
      1. Computation requirement:	O(n) for linear models; also usable for nonlinear models such as trees
      1. Pros: Smallest computational burden at training time; Enables tree-based models; Relatively easy to adapt to new categories; Handles rare categories with back-off or count-min sketch; Interpretable
      1. Cons: Requires historical data; Delayed updates required, not completely suitable for online learning; Higher potential for leakage

#### WHAT ABOUT RARE CATEGORIES?
1. One way to deal with this is through back-off, a simple technique that accumulates the counts of all rare categories in a special bin. If the count is greater than a certain threshold, then the category gets its own count statistics. Otherwise, we use the statistics from the back-off bin. This essentially reverts the statistics for a single rare category to the statistics computed on all rare categories. When using the back-off method, it helps to also add a binary indicator for whether or not the statistics come from the back-off bin.

1. count-min sketch: all the categories, rare or frequent alike, are mapped through multiple hash functions with an output range, m, much smaller than the number of categories, k. When retrieving a statistic, recompute all the hashes of the category and return the smallest statistic. Having multiple hash functions mitigates the probability of collision within a single hash function. The scheme works because the number of hash functions times m, the size of the hash table, can be made smaller than k, the number of categories, and still retain low overall collision probability.

#### GUARDING AGAINST DATA LEAKAGE
1. Since bin counting relies on historical data to generate the necessary statistics, it requires waiting through a data collection period, incurring a slight delay in the learning pipeline. Also, when the data distribution changes, the counts need to be updated. 
1. The big problem here is that the statistics involve the target variable, which is what the model tries to predict. Using the output to compute the input features leads to a pernicious problem known as leakage. 
1. If the bin-counting procedure used the current data point’s label to compute part of the input statistic, that would constitute direct leakage. One way to prevent that is by instituting strict separation between count collection (for computing bin-count statistics) and training. use an earlier batch of data points for counting, use the current data points for training (mapping categorical variables to historical statistics we just collected), and use future data points for testing. This fixes the problem of leakage, but introduces the aforementioned delay (the input statistics and therefore the model will trail behind current data).

1. It turns out that there is another solution, based on differential privacy. A statistic is approximately leakage-proof if its distribution stays roughly the same with or without any one data point. In practice, adding a small random noise with distribution Laplace(0,1) is sufficient to cover up any potential leakage from a single data point. This idea can be combined with leaving-one-out counting to formulate statistics on current data (Zhang, 2015).

#### COUNTS WITHOUT BOUNDS
1. If the statistics are updated continuously given more and more historical data, the raw counts will grow without bounds. This could be a problem for the model. A trained model “knows” the input data up to the observed scale. A trained decision tree might say, “When x is greater than 3, predict 1.” A trained linear model might say, “Multiply x by 0.7 and see if the result is greater than the global average.” These might be the correct decisions when x lies between 0 and 5. But what happens beyond that? No one knows.

1. When the input counts increase, the model will need to be retrained to adapt to the current scale. For this reason, it is often better to use normalized counts that are guaranteed to be bounded in a known interval. Another method is to take the log transform, which imposes a strict bound, but the rate of increase will be very slow when the count is very large.
Neither method will guard against shifting input distributions (e.g., last year’s Barbie dolls are now out of style and people will no longer click on those ads). The model will need to be retrained to accommodate these more fundamental changes in input data distribution, or the whole pipeline will need to move to an online learning setting where the model is continuously adapting to the input.



#### Which one to choose? 
Linear models are cheaper to train and therefore can handle noncompressed representations such as one-hot encoding. Tree-based models, on the other hand, need to do repeated searches over all features for the right split, and are thus limited to small representations such as bin counting. Feature hashing sits in between those two extremes, but with mixed reports on the resulting accuracy.


# PCA
* Model-based techniques, on the other hand, require information from the data. For example, PCA is defined around the principal axes of the data.
* Dimensionality reduction is about getting rid of “uninformative information” while retaining the crucial bits. 
* we describe the column space of a data matrix as the span of all feature vectors. If the column space is small compared to the total number of features, then most of the features are linear combinations of a few key features. Linearly dependent features are a waste of space and computation power because the information could have been encoded in much fewer features.
* The key idea here is to replace redundant features with a few new features that adequately summarize information contained in the original feature space.
* One way to mathematically define “adequately summarize information” is to say that the new data blob should retain as much of the original volume as possible. We are squashing the data blob into a flat pancake, but we want the pancake to be as big as possible in the right directions. This means we need a way to measure volume.

Volume has to do with distance. But the notion of distance in a blob of data points is somewhat fuzzy. One could measure the maximum distance between any two pairs of points, but that turns out to be a very difficult function to mathematically optimize. An alternative is to measure the average distance between pairs of points, or equivalently, the average distance between each point and its mean, which is the variance. This turns out to be much easier to optimize. (Life is hard. Statisticians have learned to take convenient shortcuts.) Mathematically, this translates into maximizing the variance of the data points in the new feature space.

## Projection
* PCA uses linear projection to transform data into the new feature space.
* The next step is to compute the variance of the projections. 
* The key lies in the sum-of-squares identity: the sum of a bunch of squared terms is equal to the squared norm of a vector whose elements are those terms, which is equivalent to the vector’s inner product with itself. 
* This formulation of PCA presents the target more clearly: we look for an input direction that maximizes the norm of the output.
* Once the principal components are found, we can transform the features using linear projection. 
* the easiest way to implement PCA is by taking the singular value decomposition of the centered data matrix.

* Due to the orthogonality constraint in the objective function, PCA transformation produces a nice side effect: the transformed features are no longer correlated. In other words, the inner products between pairs of feature vectors are zero.
* Sometimes, it is useful to also normalize the scale of the features to 1. In signal processing terms, this is known as whitening. It results in a set of features that have unit correlation with themselves and zero correlation with each other. Mathematically, whitening can done by multiplying the PCA transformation with the inverse singular values
* Whitening is independent from dimensionality reduction; one can perform one without the other. For example, zero-phase component analysis (ZCA) (Bell and Sejnowski, 1996) is a whitening transformation that is closely related to PCA, but that does not reduce the number of features. ZCA whitening uses the full set of principal components V without reduction, and includes an extra multiplication back onto VT
* When seen as a method for eliminating linear correlation, PCA is related to the concept of whitening. Its cousin, ZCA, whitens the data in an interpretable way, but does not reduce dimensionality.

* The modeling assumption here is that variance adequately represents the information contained in the data. 

## Considerations and Limitations of PCA
* When using PCA for dimensionality reduction, one must address the question of how many principal components (k) to use. Like all hyperparameters, this number can be tuned based on the quality of the resulting model. But there are also heuristics that do not involve expensive computational methods.
  * One possibility is to pick k to account for a desired proportion of total variance. The variance of the projection onto the kth component is:

║Xvk║2 =║uk σk║2 = σk2

which is the square of the kth-largest singular value of X. The ordered list of singular values of a matrix is called its spectrum. Thus, to determine how many components to use, one can perform a simple spectral analysis of the data matrix and pick the threshold that retains enough variance.
* Another method for picking k involves the intrinsic dimensionality of a dataset. This is a hazier concept, but can also be determined from the spectrum. Basically, if the spectrum contains a few large singular values and a number of tiny ones, then one can probably just harvest the largest singular values and discard the rest. Sometimes the rest of the spectrum is not tiny, but there’s a large gap between the head and the tail values. That would also be a reasonable cutoff. This method is requires visual inspection of the spectrum and hence cannot be performed as part of an automated pipeline
* One key criticism of PCA is that the transformation is fairly complex, and the results are therefore hard to interpret. The principal components and the projected vectors are real-valued and could be positive or negative. The principal components are essentially linear combinations of the (centered) rows, and the projection values are linear combinations of the columns. In a stock returns application, for instance, each factor is a linear combination of time slices of stock returns. What does that mean? It is hard to express a human-understandable reason for the learned factors. Therefore, it is hard for analysts to trust the results. If you can’t explain why you should be putting billions of other people’s money into particular stocks, you probably won’t choose to use that model.

* PCA is computationally expensive. It relies on SVD, which is an expensive procedure. To compute the full SVD of a matrix takes O(nd2 + d3) operations (Golub and Van Loan, 2012), assuming n ≥ d—i.e., there are more data points than features. Even if we only want k principal components, computing the truncated SVD (the k largest singular values and vectors) still takes O((n+d)2 k) = O(n2k) operations. This is prohibitive when there are a large number of data points or features.

* It is difficult to perform PCA in a streaming fashion, in batch updates, or from a sample of the full data. Algorithms exist, but at the cost of reduced accuracy. One implication is that one should expect lower representational accuracy when projecting test data onto principal components found in the training set. As the distribution of the data changes, one would have to recompute the principal components in the current dataset.

* Lastly, it is best not to apply PCA to raw counts (word counts, music play counts, movie viewing counts, etc.). The reason for this is that such counts often contain large outliers. As we know, PCA looks for linear correlations within the features. Correlation and variance statistics are very sensitive to large outliers; a single large number could change the statistics a lot. So, it is a good idea to first trim the data of large values , or apply a scaling transform like tf-idf or the log transform.

* PCA is very useful when the data lies in a linear subspace like a flat pancake. But what if the data forms a more complicated shape?

## Use Cases
* for small numbers of real-valued features, it is very much worth trying.
* PCA transformation discards information from the data. Thus, the downstream model may be cheaper to train, but less accurate.
* One of the coolest applications of PCA is in anomaly detection of time series. 

* The two main things to remember about PCA are its mechanism (linear projection) and objective (to maximize the variance of projected data).

# Nonlinear Featurization via K-Means Model Stacking
* A flat plane (linear subspace) can be generalized to a manifold (nonlinear subspace), which can be thought of as a surface that gets stretched and rolled in various ways. If a linear subspace is a flat sheet of paper, then a rolled up sheet of paper is a simple example of a nonlinear manifold. Informally, this is called a Swiss roll.
* Once rolled, a 2D plane occupies 3D space. Yet it is essentially still a 2D object. In other words, it has low intrinsic dimensionality, a concept we’ve already touched upon in “Intuition”. If we could somehow unroll the Swiss roll, we’d recover the 2D plane. This is the goal of nonlinear dimensionality reduction, which assumes that the manifold is simpler than the full dimension it occupies and attempts to unfold it.
* The key observation is that even when a big manifold looks complicated, the local neighborhood around each point can often be well approximated with a patch of flat surface. In other words, the patches to encode global structure using local structure. **Nonlinear dimensionality reduction** is also called **nonlinear embedding** or **manifold learning**. Nonlinear embeddings are useful for aggressively compressing high-dimensional data into low-dimensional data. They are often used for visualization in two or three dimensions. The goal of feature engineering, however, isn’t so much to make the feature dimensions as low as possible, but to arrive at the right features for the task. In this chapter, the right features are those that represent the spatial characteristics of the data.

* **Clustering algorithms** are usually not presented as techniques for **local structure learning**. But they in fact enable just that. Points that are close to each other (where “closeness” can be defined by a chosen metric) belong to the same cluster. Given a clustering, a data point can be represented by its cluster membership vector. If the number of clusters is smaller than the original number of features, then the new representation will have fewer dimensions than the original; the original data is compressed into a lower dimension. Compared to nonlinear embedding techniques, clustering may produce more features. But if the end goal is feature engineering instead of visualization, this is not a problem.

* The clustering algorithm, k-means,  performs **nonlinear manifold feature extraction**. 

## Clustering as Surface Tiling

* When data is spread out fairly uniformly (instead of clustered), there is no longer a correct number of clusters. In this case, the role of a clustering algorithm is **vector quantization**, i.e., partitioning the data into a finite number of chunks. The number of clusters can be selected based on acceptable approximation error when using quantized vectors instead of the original ones.
* Visually, this usage of k-means can be thought of  as covering the data surface with patches. This is indeed what we get if we run k-means on a Swiss roll dataset
* The problem is that if we pick a k that is too small, then the results won’t be so nice from a manifold learning perspective. Data from very different sections of the manifold being mapped to the same clusters
* If the data is distributed uniformly throughout the space, then picking the right k boils down to a **sphere-packing problem**. In d dimensions, one could fit roughly 1/r^d spheres of radius r. Each k-means cluster is a sphere, and the radius is the maximum error of representing points in that sphere with the centroid. So, if we are willing to tolerate a maximum approximation error of r per data point, then the number of clusters is O(1/r^d), where d is the dimension of the original feature space of the data.
* Uniform distribution is the worst-case scenario for k-means. If data density is not uniform, then we will be able to represent more data with fewer clusters. In general, it is difficult to tell how data is distributed in high-dimensional space. One can be conservative and pick a larger k, but it can’t be too large, because k will become the number of features for the next modeling step.

## k-Means Featurization for Classification
* When using k-means as a featurization procedure, a data point can be represented by its cluster membership (a sparse one-hot encoding of the cluster membership categorical variable
* If a target variable is also available, then we have the choice of giving that information as a hint to the clustering procedure. One way to incorporate target information is to simply include the target variable as an additional input feature to the k-means algorithm. Since the objective is to minimize the total Euclidean distance over all input dimensions, the clustering procedure will attempt to balance similarity in the target value as well as in the original feature space. The target values can be scaled to get more or less attention from the clustering algorithm. Larger differences in the target will produce clusters that pay more attention to the classification boundary.

* Clustering algorithms analyze the spatial distribution of data. Therefore, k-means featurization creates a compressed spatial index of the data which can be fed into the model in the next stage.  This is an example of **model stacking**.

## Alternative Dense Featurization
* Instead of one-hot cluster membership, a data point can also be represented by a dense vector of its inverse distance to each cluster center. One-hot cluster membership results in a very lightweight, sparse representation, but one might need a larger k to represent data of complex shapes. Inverse distance representation is dense, which could be more expensive for the modeling step, but one might be able to get away with a smaller k.

* A compromise between sparse and dense is to retain inverse distances for only p of the closest clusters. But now p is an extra hyperparameter to tune.

## Pros, Cons, and Gotchas
* Using k-means to turn spatial data into features is an example of model stacking, where the input to one model is the output of another. Another example of stacking is to use the output of a decision tree–type model (random forest or gradient boosting tree) as input to a linear classifier. 
* The key intuition with stacking is to push the nonlinearities into the features and use a very simple, usually linear model as the last layer. 
* The simple model at the top level can be quickly adapted to the changing distributions of online data. This is a great trade-off between accuracy and speed, and this strategy is often used in applications like targeted advertising that require fast adaptation to changing data distributions.

* Use sophisticated base layers (often with expensive models) to generate good (often nonlinear) features, combined with a simple and fast top-layer model. This often strikes the right balance between model accuracy and speed.

* For k-means, the training time is O(nkd) because each iteration involves computing the d-dimensional distance between every data point and every centroid (k). We optimistically assume that the number of iterations is not a function of n, though this may not be true in all cases. Prediction requires computing the distance between the new data point and each of the k centroids, which is O(kd). The storage space requirement is O(kd), for the coordinates of the k centroids.
* Logistic regression training and prediction are linear in both the number of data points and feature dimensions. RBF SVM training is expensive because it involves computing the kernel matrix for every pair of input data. RBF SVM prediction is less expensive than training; it is linear in the number of support vectors s and the feature dimension d. GBT training and prediction are linear in data size and the size of the model (t trees, each with at most 2m leaves, where m is the maximum depth of the tree). A naive implementation of kNN requires no training time at all because the training data itself is essentially the model. The cost is paid at prediction time, where the input must be evaluated against each of the original training points and partially sorted to retrieve the k closest neighbors.
* Overall, k-means + LR is the only combination that is linear (with respect to the size of training data, O(nd), and model size, O(kd)) at both training and prediction time. The complexity is most similar to that of GBT, which has costs that are linear in the number of data points, the feature dimension, and the size of the model (O(2^mt)).

* k-means featurization is useful for real-valued, bounded numeric features that form clumps of dense regions in space. The clumps can be of any shape, because we can just increase the number of clusters to approximate them. (Unlike in the classic clustering setup, we are not concerned with discovering the “true” number of clusters; we only need to cover them.)

* k-means cannot handle feature spaces where the Euclidean distance does not make sense—i.e., weirdly distributed numeric variables or categorical variables. If the feature set contains those variables, then there are several ways to handle them:

  * Apply k-means featurization only on the real-valued, bounded numeric features.
  * Define a custom metric to handle multiple data types and use the k-medoids algorithms. (k-medoids is analogous to k-means but allows for arbitrary distance metrics.)
Convert categorical variables to binning statistics (see “Bin Counting”), then featurize them using k-means.
Combined with techniques for handling categorical variables and time series, k-means featurization can be adapted to handle the kind of rich data that often appears in customer marketing and sales analytics. The resulting clusters can be thought of as user segments, which are very useful features for the next modeling step.

* This chapter illustrated the concept of model stacking using a somewhat unconventional approach: combining supervised k-means with a simple linear classifier. k-means is usually used as an unsupervised modeling method to find dense clusters of data points in feature space. Here, however, k-means is optionally given the class labels as input. This helps k-means to find clusters that better align with the boundary between classes. 

* Deep learning takes model stacking to a whole new level by layering neural networks on top of one another. Two recent winners of the ImageNet Large Scale Visual Recognition Challenge involved 13 and 22 layers of neural networks. They take advantage of the availability of lots of unlabeled training images and look for combinations of pixels that yield good image features. The technique in this chapter separately trains the k-means featurizer from the linear classifier. But it’s possible to jointly optimize the featurizer and the classier. As we shall see, deep learning training takes the latter route.

## Potential for Data Leakage
* Those who remember our caution regarding data leakage (see “Guarding against data leakage”) might ask whether including the target variable in the k-means featurization step would cause such a problem. The answer is “yes,” but not as much in the case of bin counting. If we use the same dataset for learning the clusters and building the classification model, then information about the target will have leaked into the input variables. As a result, accuracy evaluations on the training data will probably be overly optimistic, but the bias will go away when evaluating on a hold-out validation set or test set. Furthermore, the leakage will not be as bad as in the case of bin-counting statistics (see “Bin Counting”), because the lossy compression of the clustering algorithm will have abstracted away some of that information. To be extra careful about preventing leakage, hold out a separate dataset for deriving the clusters, just like in the case of bin counting.


# Automating the Featurizer: Image Feature Extraction and Deep Learning
* Machine learning models require semantically meaningful features to make semantically meaningful predictions. 
## The Simplest Image Features (and Why They Don’t Work)
* color information is probably not enough to characterize an image.
* Another simple idea is to measure the pixel value differences between images, but it is too stringent as a similarity measure. 
* The problem is that individual pixels do not carry enough semantic information about the image. Therefore, they are bad atomic units for analysis.

## Manual Feature Extraction: SIFT (Scale Invariant Feature Transform) and HOG (Histogram of Oriented Gradients)
* Both of them essentially compute histograms of gradient orientations.

### Image Gradients
* The difference in value between neighboring pixels is called an image gradient. This involves two 1D difference operations that can be handily represented by a vector mask or filter. 
* To apply a filter to an image, we perform a convolution. It involves flipping the filter and taking the inner product with a small patch of the image, then moving to the next patch.
* The horizontal gradient picks out strong vertical patterns such as the inner edges of the cat’s eyes, while the vertical gradient picks out strong horizontal patterns such as the whiskers and the upper and lower lids of the eyes. This might seem a little paradoxical at first, but it makes sense once we think about it a bit more. The horizontal (x) gradient identifies changes in the horizontal direction. A strong vertical pattern spans multiple y pixels at roughly the same x position. Hence, vertical patterns result in horizontal differences in pixel values. This is what our eyes detect as well. 

### Gradient Orientation Histograms
* How can we summarize the image gradients in a neighborhood? A statistician would answer, “Look at the distribution!” SIFT and HOG both take this path. In particular, they compute (normalized) histograms of the gradient vectors as image features.
* SIFT and HOG settled on a scheme where the image gradients are binned by their orientation angle θ, weighted by the magnitude of each gradient. 

* HOG and SIFT both settled on a two-level representation  of image neighborhoods: first adjacent pixels are organized into cells, and neighboring cells are then organized into blocks. An orientation histogram is computed for each cell, and the cell histogram vectors are concatenated to form the final feature descriptor for the whole block.

## Learning Image Features with Deep Neural Networks
* The convolution operator captures the effect of a linear system, which multiplies the incoming signal with its response function, summing over current responses to all past input.








# References
1. Feature Engineering for Machine Learning, by Alice Zheng; Amanda Casari, 2018
1. Feature Engineering Made Easy, by Sinan Ozdemir; Divya Susarla, 2018
