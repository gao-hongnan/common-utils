# AISG Training Chronicles

See https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html

---

I think it really depends on our loss function $\mathcal{L}$ or equivalently the
cost function $\mathcal{J}$ if you prefer.

- **Case 1**: $\mathcal{L}(\boldsymbol{\theta})$ is convex over $\Theta$ (note
    the emphasis that the loss is a function of the parameters, not the data),
    where $\boldsymbol{\theta} \in \Theta \subseteq \mathbb{R}^D$.

  - $\mathcal{L}$ has a unique global minimum $\boldsymbol{\theta}^*$ in
        $\Theta$:
        $$
        \exists \boldsymbol{\theta}^*\in \Theta, \forall \boldsymbol{\theta} \in \Theta, \mathcal{L}(\boldsymbol{\theta}^*) \leq \mathcal{L}(\boldsymbol{\theta})
        $$
        where $D$ is the dimension of the parameter space (we are being slighly
        less pedantic here as we are not specifying the topology of the
        parameter space, but let's assume this parameter is a flattened vector
        of all the parameters of the model).

  - Optimization algorithms such as gradient descent can be employed to find
        the global minimum $\boldsymbol{\theta}^*$ in $\Theta$ that minimizes
        $\mathcal{L}$.

  - Any local minimum $\boldsymbol{\theta}^*$ in $\Theta$ is also the global
        minimum $\boldsymbol{\theta}^*$ in $\Theta$.

  - Given an appropriate learning rate $\eta$, the negative gradient of
        $\mathcal{L}$ always points in the direction of the steepest descent in
        $\Theta$. Hence, gradient-based algorithms are guaranteed to converge to
        the global minimum $\boldsymbol{\theta}^*$ when $\mathcal{L}$ is convex
        over $\Theta$.

---

- **Case 2**: $\mathcal{L}(\boldsymbol{\theta})$ for deep neural networks over
    $\Theta$, where $\boldsymbol{\theta} \in \Theta \subseteq \mathbb{R}^D$.

  - **Non-convexity**: Unlike simple models where the loss might be convex,
        the loss landscape of deep neural networks is typically non-convex. This non-convexity can lead to multiple minima (all eigenvalues of the loss function’s Hessian at zero gradient > 0) and saddle points (where some eigenvalues of the Hessian are positive and some are negative).

  - **Local Minima and Saddle Points**: While there may be many local
        minima, recent research suggests that in high-dimensional spaces (like
        those of deep nets), saddle points are more prevalent. At a saddle
        point, the gradient is zero, but it's neither a minimum nor a maximum.

  - **Optimization Algorithms**: Gradient-based methods, like gradient
        descent and its variants (e.g., SGD, Adam), are commonly used. While
        these methods are not guaranteed to find the global minimum due to the
        non-convex nature of the loss, they are often effective at finding "good
        enough" local minima.

So I think that is why researchers often empirically observe that deep neural
networks "converge" when the loss curves start to flatten out. But one thing to
distinguish is that convergence is not the same as generalization. The former



### Is if there exist any theoretical bound of the loss function $\mathcal{L}$?

so at least we know what is impossible to achieve below.


1. **Theoretical Bounds Based on Data**: In an idealized setting, if your data were noise-free and the neural network had the capacity to represent the underlying function perfectly, then the training loss could, in theory, be zero. However, in real-world scenarios with noisy data or inherent ambiguities, the lower bound on the loss might be greater than zero. This is especially true for regression tasks where the noise in the data sets a floor on how low the loss can go.

2. **Empirical Bounds**: In practice, the best way to determine a realistic lower bound is empirically, by training various models on your data and observing the lowest loss achieved. Over time, as you experiment with different architectures, regularization methods, and training strategies, you can get a sense of what a good lower bound for your specific problem and dataset might be.

3. **Generalization Gap**: It's worth noting that even if the training loss is very low, the validation or test loss might be higher due to overfitting. The difference between training and validation loss is referred to as the "generalization gap." A model that has a very low training loss but a significantly higher validation loss may not be as useful as one with a slightly higher training loss but a smaller generalization gap.

4. **The Complexity of Neural Network Loss Landscapes**: Deep neural networks have a highly non-convex loss landscape. While there might be many local minima, recent research suggests that many of these minima are surrounded by flat regions (often referred to as "plateaus") and that these different minima might have very similar loss values. This makes determining a strict lower bound challenging.

5. **Dependence on the Loss Function**: The lower bound is also intrinsically tied to the loss function being used. For example, a mean squared error (MSE) loss can theoretically range from 0 to positive infinity, while a binary cross-entropy loss for a well-calibrated model can range between 0 and 1 (in the case of perfect classification and absolute misclassification, respectively).

6. **Problem-Specific Bounds**: For certain problems, especially those with well-defined constraints, it might be possible to derive theoretical bounds. For instance, in some physical modeling problems where the underlying dynamics are well-understood, one might be able to derive theoretical bounds on the achievable loss. However, such cases are more the exception than the rule in deep learning.

In summary, while there isn't a universal lower bound for the loss value in deep neural networks that applies across all scenarios, understanding the specifics of a given problem, dataset, and model can provide insights into what a reasonable lower bound might be.

---

### How to link to poor convergence?

1. **Caveats**:
    - **Learning Rate**: The choice of learning rate is crucial. If it's too
      large, gradient descent can oscillate around the minimum or even diverge.
      If it's too small, convergence can be very slow. For convex problems,
      there are theoretical bounds on the learning rate to ensure convergence.
    - **Convergence to Global Minimum**: While gradient descent is guaranteed to
      converge to the global minimum for convex functions, the convergence might
      be slow, especially if the function is poorly conditioned or if the
      learning rate is not well-tuned.
    - **Noise and Stochasticity**: In the context of machine learning, we often
      use Stochastic Gradient Descent (SGD) or its variants, which estimate the
      gradient using a subset of the data. This introduces noise into the
      gradient updates, which can cause oscillations. However, on average, the
      method still moves towards the global minimum for convex functions.