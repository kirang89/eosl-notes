# Chapter 1: Introduction

*"There is no true interpretation of anything; interpretation is a vehicle in the service of human comprehension. The value of interpretation is in enabling others to fruitfully think about an idea."* 

– Andreas Buja

# Chapter 2: Overview of Supervised Learning

In the statistical literature the inputs are often called the predictors, a
term we will use interchangeably with inputs, and more classically the
independent variables. In the pattern recognition literature the term features
is preferred, which we use as well. The outputs are called the responses, or
classically the dependent variables.



Input Variable Types:
- Quantitative (numeric)

- Qualitative (text)

- Ordered Categorical (eg: small, medium, large i.e no appropriate metric
  notion)

  ​

Multiple qualitative labels can be encoded using *dummy variables*. A K-level
variable is representated using K-level binary variables, only one of which is
active at any time.



## Linear Regression Model

The Linear Model for prediction is represented as:
$$
\hat{Y} = \beta_0 + \sum_{j=1}^p X_j \beta_j
$$

or in general form as:
$$
\hat{Y} = X^T \hat{\beta}
$$
The term $\beta_0$ is the intercept, also known as the **bias** in machine learning. 		

The linear model makes huge assumptions about structure and yields stable but possibly   		inaccurate predictions. The method of k-nearest neighbors makes very mild structural assumptions: its predictions are often accurate but can be unstable.

The important methods for fitting a linear model are:

- Least Squares
- Nearest Neigbours

### Least Squares

In this approach, we pick coefficients of $\beta$ that minimize the *residual sum of squares*:
$$
RSS(\beta) = \sum_{i=1}^N (y_i - x_i^T\beta)^2
$$
or in matrix notation:
$$
RSS(\beta) = (y - X\beta)^T(y-X\beta)
$$
Differentiating w.r.t $\beta$ , we get
$$
X^T (y - X\beta) = 0
$$
If $X^TX$ is nonsingular, the unique solution is:
$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$
------

*The estimator is unbiased and consistent if the errors have finite variance and is unrelated to the regressors.*

------

This means, for an arbitrary input $x_0$ the prediction is
$$
\hat{\textbf{y}}(\textbf{x}_0) = x_0^T \hat{\beta}
$$



------

*The least squares approach can be used to fit models that are not linear models. Thus, although the terms "least squares" and "linear model" are closely linked, they are not synonymous.*

------



#### Gaussian (or Normal) Distribution

- Developed for approximating the binomial distribution

- Normal distributions have many convenient properties, so random variates with unknown distributions are often assumed to be normal, especially in physics and astronomy. Although this can be a dangerous assumption, it is often a good approximation due to a surprising result known as the **central limit theorem**. *This theorem states that the mean  of any set of variates with any distribution having a finite mean and variance tends to the normal distribution.* Many common attributes such as test scores, heights of people, errors in measurement etc., follow roughly normal distributions, with few members at the high and low ends and many in the middle.

  ​

  ![Example of a Gaussian Curve](http://introcs.cs.princeton.edu/java/11gaussian/images/stddev.png)

### Nearest Neighbours

This approach uses *k* closest observations in the input space X to determine $\hat{Y}$ 
$$
\hat{Y} = \frac{1} {k} \sum_{x_i \in N_k(x)} y_i
$$
where $N_k(x)$ is the neighbourhood of x defined by k closest points in the training sample. In other words, we find k closest observations of $x_i$ in $x$ and average their responses.

​	*A large subset of the most popular techniques in use today are variants of these two simple procedures. In fact 1-nearest-neighbor, the simplest of all, captures a large percentage of the market for low-dimensional problems.*



## Statistical Decision Theory

We seek a function $f(x)$ for predicting Y given input X. This theory requires a **loss function** $L(Y, f(x))$ for penalizing errors in prediction, and by far the most convenient method is the **squared error loss** $L(Y, f(X)) = (Y  - f(X))^2$. The **Expected (Squared)  Prediction Error**, a criteria for determining $f$ is
$$
EPE(f) = E(Y - f(X))^2  \\
\hspace{4 cm} = \int [y - f(x)]^2 Pr(dx, dy)
$$
where $E(Y - f(X))^2 $ is the $L2$ loss function . The conditional expectation $f(x)$ is determined using:
$$
f(x) = E(Y | X = x)
$$
This is also know as the **regression function**. Thus the best prediction of $Y$ at any point $X=x$ is the **conditional mean**, when best is measured by **average squared error**.

If we replace the $L2$ loss function with $L1: E(Y - f(X))$, we get the **conditional median** 
$$
\hat{f} = median(Y|X=x)
$$
Although the estimates are more robust than that of the conditional mean, the $L1$ derivatives have discontinuities in their derivatives which have hindered their widespread use.



​	When the output is a categorical variable $G$, we use a **zero-one** loss function where all misclassifications are charged a single unit. The Expected Prediction Error is given by
$$
EPE = E[L(G, \hat{G}(X))]
$$
where $\hat{G}$ is the estimator. Our loss function can be represented by a $K ×K$ matrix $L$, where $K = card(G)$. $L$ will be zero on the diagonal and nonnegative elsewhere, where $L(k, ℓ)$ is the price paid for classifying an observation belonging to class $G_k$ as $G_ℓ$. On further simplification of the above, we get 
$$
\hat{G}(x) = G_k \hspace{0.3cm}if \hspace{0.1cm}Pr(G_k|X=x)  =  max_{g \in G} Pr(g | X=x)
$$
This solution is known as the **Bayes Classifier**, where we classify to the most probable class using the probability distribution $Pr(G|X)$. 



The complexity of functions of many variables can grow exponentially with the dimension, and if we wish to be able to estimate such functions with the same accuracy as function in low dimensions, then we need the size of our training set to grow exponentially as well.



The nearest-neighbour model can be viewed as a direct estimate of the regression function $f(x) = E(Y|X = x)$ but it can fail in the following cases:

- if the dimension of the input space is high, the nearest neighbors need not be close to the target point, and can result in large errors
- if special structure is known to exist, this can be used to reduce both the bias and the variance of the estimates.

In such cases, an **additive model** can be useful approximation of the truth. Additive models are typically not used for qualitative outputs.



## The function-fitting paradigm from a machine learning point of view

Suppose for simplicity that the errors are additive and that the model  $Y = f(X)+ε$  is a reasonable assumption. Supervised learning attempts to learn $f$ by example through a teacher. One observes the system under study, both the inputs and outputs, and assembles a training set of observations $T = (x_i, y_i), i = 1, . . . ,N$. The observed input values to the system $x_i$ are also fed into an artificial system, known as a *learning algorithm* (usually a computer program), which also produces outputs $\hat{f}(x_i)$ in response to the inputs. The learning algorithm has the property that it can modify its input/output relationship $\hat{f}$ in response to differences $y_i −  \hat{f}(x_i)$ between the original and generated outputs. This process is known as **learning by example**. Upon completion of the learning process the hope is that the artificial and real outputs will be close enough to be useful for all sets of inputs likely to be encountered in practice.



## The function-fitting paradigm from a mathematical & statistical point of view

The approach taken in mathematics and statistics has been from a perspective of *function approximation and estimation*. Here the data pairs ${x_i, y_i}$ are viewed as points in a (p+1)-dimensional Euclidean space. Treating it this way encourages the use of geometrical concepts of Euclidean space and mathematical concepts of probabilistic inference to be applied to the problem.



## Maximum-Likelihood Estimation

While least squares is generally very convenient, it is not the only crite- rion used and in some cases would not make much sense. A more general principle for estimation is *maximum likelihood estimation*. Suppose we have a random sample $y_i, i = 1, . . . ,N$ from a density $Pr_θ(y)$ indexed by some parameters $\theta$ *(many approximations have associated a set of parameters $\theta$ that can be modified to suit the data at hand)*. The log-probability of the observed sample is
$$
L(\theta) = \sum_{i=1}^N log Pr_\theta(y_i)
$$
The principle of maximum-likelihood assumes that the most reasonable values for $\theta$ are the ones for which the log-probability of the observed sample is the largest. 



## The Need for More Structured Regression Models

Consider the RSS criterion for a arbitrary function $f$
$$
RSS(f) = \sum_{i=1}^N (y_i - f(x_i))^2
$$
Minimizing the above leads to infinitely many solutions: any function $\hat{f}$ passing through the training points $(x_i, y_i)$ is a solution. Any particular solution chosen might be a poor predictor at test points different from the training points.

​	In order to obtain useful results for finite $N$, we must restrict the eligible solutions to the above to a smaller set of functions. How to decide on the nature of the restrictions is based on considerations outside of the data. These restrictions are sometimes encoded via the parametric representation of $f_θ$, or may be built into the learning method itself, either implicitly or explicitly.

​	In general the constraints imposed by most learning methods can be described as *complexity restrictions* of one kind or another. This usually means some kind of regular behavior in small neighborhoods of the input space. That is, for all input points $x$ sufficiently close to each other in some metric, $\hat{f}$ exhibits some special structure such as nearly constant, linear or low-order polynomial behavior. The estimator is then obtained by averaging or polynomial fitting in that neighborhood

​	In Summary, any method that attempts to produce locally varying functions in small isotropic neighborhoods will run into problems in high dimensions. And conversely, all methods that overcome the dimensionality problems have an associated—and often implicit or adaptive—metric for measuring neighborhoods, which basically does not allow the neighborhood to be simultaneously small in all directions.

​	The wide variety of learning method fall into different classes depending on the type of restrictions imposed. The three broad classes are:

- Roughness Penalty and Bayesian Methods
- Kernel Methods and Local Regression
- Basis Functions and Regression Methods



## Model Selection and Bias-Variance Tradeoff

All the models described above and many others discussed in later chapters have a smoothing or complexity parameter that has to be determined:

- the multiplier of the penalty term; 
- the width of the kernel; 
- or the number of basis functions.





![test and training error as a function of model complexity](http://3.bp.blogspot.com/-I7RePzkTwV8/Uv3juzHvmhI/AAAAAAAABls/AiyXsLCpE8o/s1600/Screen+Shot+2014-02-14+at+11.35.49+AM.png)



The training error tends to decrease whenever we increase the model complexity, that is, whenever we fit the data harder. However with too much fitting, the model adapts itself too closely to the training data, and will not generalize well (i.e., have large test error). In that case the predictions $\hat{f}(x_0)$ will have large variance. In contrast, if the model is not complex enough, it will *underfit* and may have large bias, again resulting in poor generalization. 