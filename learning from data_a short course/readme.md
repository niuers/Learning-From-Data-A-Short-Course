
## The complexity of $\mathcal{H}$
* The number of hypothesis functions $M$ can be thought of as a 'complexity' measure of the hypothesis set $\mathcal{H}$.
* If $M$ goes up, then $e_{in}(g)$ will be a poor estimator of $e_{out}(g)$ according to the inequality (hence helps question 1), however there'll be a better chance to get small $e_{in}(g)$ (hence hurts question 2).

## The complexity of the target function $f$
* The complexity of $f$ doesn't affect how well  $e_{in}(g)$ approximates  $e_{out}(g)$
* However, we'll get a worse  $e_{in}(g)$ when $f$ is complex.

* As long as we make sure that the complexity of $\mathcal{H}$ gives us a good Hoeffding bound, our success or failure in learning $f$ can be determined by our success or failure in fitting the training data.

# 1.4 Error and Noise
## 1.4.1 Error Measure
* An error measure quantifies how well each hypothesis function $h$ in the model approximates the target function $f$.
$Error = E(h,f)$.
* One may view $E(h,f)$ as the cost of using $h$ when you should use $f$.
* The choice of the error measure affects the outcome of learning process.
* The choice of the error measure depends on how the system is going to be used, rather than on any inherent criterion that we can independently determine during learning process.
