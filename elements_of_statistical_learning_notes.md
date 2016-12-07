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

*Linear regression is a statistical procedure for predicting the value of a dependent variable from an independent variable when the relationship between the variables can be described with a linear model.*

The Linear Model for prediction is represented as:
$$
\hat{Y} = \beta_0 + \sum_{j=1}^p X_j \beta_j
$$

or in general form as:
$$
\hat{Y} = X^T \hat{\beta}
$$
The term $\beta_0$ is the intercept, also known as the **bias** in machine learning. 		



------

**Why should we assume that the effects of different independent variables(X) on the expected value of the dependent variable(Y) are additive?** *This is a very strong assumption, stronger than most people realize.  It implies that the marginal effect of one independent variable  does not depend on the current values of other independent variables.  But… why shouldn’t it?  It’s conceivable that one independent variable could amplify the effect of another, or that its effect might vary systematically over time.*

------



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

- Normal distributions have many convenient properties, so random variates with unknown distributions are often assumed to be normal, especially in physics and astronomy. Although this can be a dangerous assumption, it is often a good approximation due to a surprising result known as the **Central Limit Theorem**. *The theorem states that the sum or average of a sufficiently large number of independent random variables–whatever their individual distributions–approaches a normal distribution.* Many common attributes such as test scores, heights of people, errors in measurement etc., follow roughly normal distributions, with few members at the high and low ends and many in the middle.

  ​

  ![Example of a Gaussian Curve](http://introcs.cs.princeton.edu/java/11gaussian/images/stddev.png)




------

#### An Estimation Example using the Least Squares method

As a result of an experiment, four  $(x,y)$ data points were obtained,$(1,6), (2,5), (3,7), (4, 10)$.  We hope to find a line  $y=\beta _{1}+\beta _{2}x$ that best fits these four points. In other words, we would like to find the numbers $\beta_1$ and $\beta_2$ that approximately solve the overdetermined linear system of four equations in two unknowns in some "best" sense.
$$
\beta _{1}+1\beta _{2}= 6 \\ \beta _{1}+2\beta _{2} =5 \\ \beta _{1}+3\beta _{2}=7 \\ \beta _{1}+4\beta _{2}= 10
$$
​	The "error", at each point, between the curve fit and the data is the difference between the 	right- and left-hand sides of the equations above. The least squares approach to solving this problem is to try to make the sum of the squares of these errors as small as possible; that is, to find the minimum of the function
$$
S(\beta _{1},\beta _{2}) = \left[6-(\beta _{1}+1\beta _{2})\right]^{2} + \left[5-(\beta _{1}+2\beta _{2})\right]^{2} + \left[7-(\beta _{1}+3\beta _{2})\right]^{2} + \left[10-(\beta _{1}+4\beta _{2})\right]^{2} \\= 4\beta _{1}^{2} + 30\beta _{2}^{2} + 20\beta _{1}\beta _{2} - 56\beta _{1} - 154\beta _{2} + 210
$$


The minimum is determined by calculating the [partial derivatives](https://www.wikiwand.com/en/Partial_derivative) of ![S(\beta _{1},\beta _{2})](https://wikimedia.org/api/rest_v1/media/math/render/svg/6aee70482be7faa52d3aead2d4f0987d3f92f171) with respect to ![\beta _{1}](https://wikimedia.org/api/rest_v1/media/math/render/svg/eeeccd8b585b819e38f9c1fe5e9816a3ea01804c) and ![\beta _{2}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8d30285b40d7488ae6caef3beb7106142869fbea) and setting them to zero
$$
{\frac {\partial S}{\partial \beta _{1}}} = 0 = 8\beta _{1}+20\beta _{2}-56 \\

\hspace{0.5cm}{\frac {\partial S}{\partial \beta _{2}}} = 0 =20\beta _{1}+60\beta _{2}-154.
$$


This results in a system of two equations in two unknowns, called the normal equations, which give, when solved
$$
\beta_{1} = 3.5\\
\beta_{2} = 1.4
$$
and the equation ![y=3.5+1.4x](https://wikimedia.org/api/rest_v1/media/math/render/svg/90f80928662ab155de21f253fbd1512808986d5d) of the line of best fit. The [residuals](https://www.wikiwand.com/en/Residual_(statistics)), that is, the discrepancies between the ![y](https://wikimedia.org/api/rest_v1/media/math/render/svg/b8a6208ec717213d4317e666f1ae872e00620a0d) values from the experiment and the ![y](https://wikimedia.org/api/rest_v1/media/math/render/svg/b8a6208ec717213d4317e666f1ae872e00620a0d) values calculated using the line of best fit are then found to be ![1.1,](https://wikimedia.org/api/rest_v1/media/math/render/svg/0f2ee9cbd21913f338465626f4acddc2c265e4bc) ![-1.3,](https://wikimedia.org/api/rest_v1/media/math/render/svg/5d283bb344d339f0ea3aee2498e69a1dd84f4a9a) ![-0.7,](https://wikimedia.org/api/rest_v1/media/math/render/svg/7c510c19eeb5a4c12d39b57069b867713b42c245) and ![0.9](https://wikimedia.org/api/rest_v1/media/math/render/svg/12fb1a4ae271b93c61fe117b691677bb27609f25) (see the picture on the right). The minimum value of the sum of squares of the residuals is ![S(3.5,1.4)=1.1^{2}+(-1.3)^{2}+(-0.7)^{2}+0.9^{2}=4.2.](https://wikimedia.org/api/rest_v1/media/math/render/svg/a74f0a3448ba5046b3c39120cc52154b211e39e9)

------



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



# Chapter 3: Linear Methods for Regression

*Linear models* were largely developed in the precomputer age of statistics. For prediction purposes they can sometimes outperform fancier nonlinear models, especially in situations with small numbers of training cases, low signal-to-noise ratio or sparse data. Finally, linear methods can be applied to transformations of the inputs and this considerably expands their scope. These generalizations are sometimes called *basis-function* methods.



------

*Many non-linear techniques are direct generalizations of the linear methods.*

------



As seen in Chapter 2, the predicted values for the input vector using the linear model is given by
$$
\hat{y} = X\hat{\beta} = X (X^TX)^{-1} X^Ty
$$
The matrix $H = X (X^TX)^{-1} X^T$ is called the **hat matrix**. 



![least squares estimation](https://4.bp.blogspot.com/-wQwYTZR1cU0/Uv41AN0MZII/AAAAAAAABnQ/jRtYqBvKB7Y/s1600/Screen+Shot+2014-02-14+at+5.21.38+PM.png)

We minimize $RSS(β) = (y −Xβ^2)$ by choosing $\hat{\beta}$ so that the residual vector $y − \hat{y}$ is orthogonal to this subspace. This orthogonality is expressed in the figure above, and the resulting estimate $\hat{y}$ is hence the **orthogonal projection** of $y$ onto this subspace. The hat matrix $H$ computes the orthogonal projection, and hence it is also known as a **projection matrix**.

​	It might happen that the columns of $X$ are not linearly independent, so that $X$ is not of full rank. This would occur, for example, if two of the inputs were perfectly correlated, (e.g.,$ x_2 = 3 x_1$). Then $X^TX$ is singular and the least squares coefficients  $\hat{\beta}$ are not uniquely defined. However, the fitted values $\hat{y} = X \hat{\beta}$ are still the projection of $y$ onto the column space of $X$; there is just more than one way to express that projection in terms of the column vectors of $X$. The non-fullrank case occurs most often when one or more qualitative inputs are coded in a *redundant* fashion. There is usually a natural way to resolve the non-unique representation, by *recoding and/or dropping redundant columns in $X$*.



A measure of the absolute amount of variability in a variable is (naturally) its **variance**, which is defined as its *average squared deviation from its own mean*.



In using *linear* models for prediction, it turns out very conveniently that the *only* statistics of interest (at least for purposes of estimating coefficients to minimize squared error) are the **mean** and **variance** of each variable and the **correlation coefficient** between each pair of variables. The coefficient of correlation between X and Y is commonly denoted by $r_{XY}$, and it measures the strength of the linear relationship between them on a relative (i.e., unitless) scale of $-1$ to $+1$. That is, it measures the extent to which a linear model can be used to predict the deviation of one variable from its mean given knowledge of the other's deviation from its mean at the same point in time.