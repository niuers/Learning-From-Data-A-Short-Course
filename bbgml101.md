Ivanov and Tikhonov are equivalent: (Proof) HW4 2018

09/30/19:
SVM:
1. True
2. False
3. (a) False, need \alpha_i
(b) True

Does SVM model assume linear separable?
only hard margin SVM assumes this, not soft margin SVM.

10/01/19:
Subgradients
Q1: slide 47, what does the 'best' in x^{k}_{best} mean here? It's the closest solution in the iterations but may not be the last step.

10/19:
Slide Self attention: \sqrt(dt) o reduce sentivity of softmax

Q2: Do we only have subgradients on convex functions? 

Quiz:
1. GD will find minima for convex and differentiable functions, not sure about differentiable functions only? What's the theorem for this?
For Subgradient descent, when iterations go to infinity, the limit is the minima.

2. Easy
3. for any x, y \in a sublevel set C of convext function, f(x) <= c, f(y) <= c, 
f[(1-t)x + ty] <= (1-t)f(x) + tf(y) <= c
4. have a function's domain on 5 <= x1^2 + x2^2 <= 10
5. f(x) = |x|, f(x) = 1/x^2
6. If f is differentiable and convex, and grad(f) = 0, then x is the global minimizer. If f is not differentiable, but 
0 belongs to subdifferential of f, then x is a global minimizer of f.

7. for GD, g is gradient, for sub-gd, g is subgradient
8 (a) No. G can be large. Could be bouncing around. What does convengence mean? 

Connect math formula with pictures. Connect math with algorithms(into a setting). Connect algorithms with pictures. What about library's algorithm? 
Given a concept, how do we compute it with example? how do we use it? Is there a general algorithm for the computation? If so, why it's still useful? How does it compare to other methods? 

Proof of theorem in the last two slides.

 
For gradient descent with fixed step size, we get actual convergence as long as the step size is small

What's the variable name correspond to in machine learning algorithm? x  is weight.

Kernel Method:
1. If we standardize the features, do we sill see the effect of coefficients in kernel? 
2. 

10/07/19: Featurization
10/08/19: Kernel
1.  A norm definition: ||x|| = sqrt(2*x_1^2 + x_2)
An inner product: <x, y> = x \dot y = x_1y_1 + x_2y_2

What if I have two examples with slightly different numbers?
What are the most likely values of some parameters? 
How do people use the theorem? algorithm?
What are the disadvantages of the algorithm?
In addition to the examples you give, (which might be what we know), what are other examples that this algorithm can be applied?

They don't correspond to the relationship: ||x|| = sqrt(x_1^2+x_2^2)
If something doesn't work for some cases, could you give concret examples? 

10/14/19:
Stratification: less bias more variance
Binning: more bias less variance

histogram vs. Gamma: 

Equivalence of Lagrange duality
min x + 3y, x + y \ge 2, x \ge 0, y \ge 0

min px + qy, x + y \ge 2, x \ge 0, y \ge 0

a(x+y) \ge 2a
bx \ge 0
cy \ge 0

(a+b)x + (a+c)y \ge 2a 

suppose we have a + b = p, a + c = q, we have px+qy \ge 2a
The maximum of a satisfies :
a+b = p, a + c = q, and a,b,c \ge 0

This is equivalent to the original probelm.
what if p is negative, the original problem has minimum of -infty, the new problem has no solution for 'max 2a', so the solution is -infty.


Linear problems always have strong duality

Slater's condition

1. Subgradients of a sum of functions: x + |x|

10/21/19:
1. Can think 1 = exp(0^Tx)
2. Matrix is just transformation
What is the determinant of matrix? How much volume changes from one dimension to another dimension.
