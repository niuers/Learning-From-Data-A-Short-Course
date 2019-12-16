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

