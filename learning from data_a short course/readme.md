# Chapter 1
#### Exercise 1.4
Pick line $y=2x+2$ as the separating line in the 2D plane. We need pick 20 data points:

x1, x2, y
0,0,1
1,1,1
0,1,1
-0.5,0,1
-1,-1,1
2,4,1
1,-1,1
3,-5,1
0.5,-2,1
-1,1,-1
0,3,-1
-2,0,-1
-3,-1,-1
1,6,-1
2,8,-1
-4,2,-1
-3,3,-1
3,10,-1
-0.5,2,-1
-2,1,-1



# Chapter 2 Traing versus Testing
We need a mathematical theory to characterize the distinction between training and testing. 

## 2.1 Theory of Generalization

The out-of-sample error $E_{out}$ measures how well our training on data $\mathcal{D}$ has generalized to data we haven't seen before. It is based on the performance over the entire input space $\mathcal{X}$. 

* Generalization Error
  * One can define the generation error as the discrepancy between $E_{in}$ and $E_{out}$. The Hoeffding inequality provides a way to characterize the generation error with a probablistic bound.
* Generalization Bound: it bounds $E_{out}$ in terms of $E_{in}$.
  * From Hoeffding inequality, we can derive two generation bounds for $E_{out}$. For a given tolerance $\delta$, with probability of at least $1-\delta$, and for the final hypothesis $g$ with minimum $E_{in}$, we have
  \begin{align}
  E_{out}(g) &\le E_{in}(g) + \sqrt{\frac{1}{2N}ln\frac{2M}{2\delta}}
  E_{out}(g) &\ge E_{in}(g) - \sqrt{\frac{1}{2N}ln\frac{2M}{2\delta}}
  \end{align}
  
  * The first bound ensures that the final hypothesis $g$ will do well in out-of-sample. 
  * The second bound ensures that we did the best we could we our $\mathcal{H}$. No other hypothesis $h\in \mathcal{H}$ has $E_{out}(h)$ significantly better than $E_{out}(g)$ because every hypothesis with a higher $E_{in}$ than $g$ will have a comparably higher $E_{out}$ due to the second bound.

* The error bound however, can go infinite with the number of hypothesis $M$. We need to replace $M$ with some other meaningful value.
* Notice that the error bound is derived using union bounds, while in reality, many hypothesis $h$ can have large overlap, thus the probability of unioned events is much smaller than indicated by the union bound. 
  
### 2.1.1 Effective Number of Hypotheses

* Growth function: It is a combinatorial quantity that captures how difference the hypotheses in $\mathcal{H}$ are, and hence how much overlap the different events have. It is based on the number of different hypotheses that $\mathcal{H}$ can implement but only on a finite sample of points rather than the entire input space $\mathcal{X}$.  
* We are going to replace the $M$ in bound by growth function.

## 2.2 Interpretataion of the Generalization Bound
* The VC analysis establishes the feasibility of learning for infinite hypothesis set, the only kind we use in practice.
* Although the bound is loose, it tends to be equally loose for different learning models, and hence is useful for comparing the generalization performance of these models.

* Some practical Observations
  * Learning models with lower $d_{VC}$ tend to generalize better than those with higher $d_{VC}$.
  * A popular rule of thumb to get a decent generalization performance: requires the number of samples is approximately proportional to the VC dimension, for example $N \ge 10d_{VC}$.

### 2.2.1 Sample Complexity
* The sample complexity determines how many training examples $N$ are needed to achieve a certain generalization performance.

### 2.2.2 Penalty for Model Complexity
* From VC generalization bound, we have 
\begin{align}
E_{out}(g) &\le E_{in}(g) + \Omega(N, \mathcal{H}, \delta)\\
\end{align}
* The bound $\Omega(N, \mathcal{H}, \delta)$ can be treated as a penalty of model complexity. 
  * If we use more complex $\mathcal{H}$, i.e. larger $d_{VC}$, it's larger.
  * If we insist higher confidence, i.e. lower $\delta$, it's larger.
  * If we have more training examples $N$, it's smaller.

* $E_{in}(g)$
  * If we use more complex $\mathcal{H}$, $E_{in}(g)$ is smaller

* There's tradeoff between $\Omega$ and $E_{in}$. The optimal model is a compromise that minimizes a combination of the two terms.

### 2.2.3 The Test Set
* While the $E_{in}$ helps in training process and gives us a loose estimate (generation bound) of $E_{out}$, it is useless If our goal is to get an accurate forecast of $E_{out}$.
* If we use a test set and compute $E_{test}$, it is a better approximation to $E_{out}$. Here we can use Hoeffding inequality for a single hypothesis since the final hypothesis $g$ has been fixed. 
* The test set is also not biased in its estimate of $E_{out}$. The training set has an optimistic bias since it was used to choose a hypothesis that look good on it. The test set just has straight finite-sample variance (so does training set) but no bias. 

### 2.2.4 Other Target Types
* We can just change the definition of $E_{in}$ using expected squared errors.

## 2.3 Approximation-Generalization Tradeoff
* When we select the hypothesis set $\mathcal{H}$, we need balance two conflicting goals: The VC bound is one way to look at this tradeoff (especially for binary error).
  * To have some hypothesis in $\mathcal{H}$ that can approximate $f$. If $\mathcal{H}$ is too simple, we fail to approximate $f$ well, and end up with a large in-sample error term.
  * To enable the data to zoom in on the right hypothesis. If $\mathcal{H}$ is too complex, we may fail to generalize well because of the large model complexity term. 

* Bias and Variance is another way to look at the tradeoff, which is particularly suited for squared error. 

### 2.3.1 Bias and Variance
* Variance: one can view it as a measure of 'instability' in the learning model.




  
