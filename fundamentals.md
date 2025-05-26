class: middle, center, title-slide

# Differentiable programming

Lecture 1: Fundamentals

<br><br>
Mathieu Blondel, Vincent Roulet

---

# Outline

- Differentiation
  * Continuous functions
  * Differentiable functions
  * Gradients
  * Jacobians
  * Linear maps: JVPs and VJPs
  * Hessians
- Probabilistic learning

---

## Continuous functions

A function
$f:\mathbb{R} \rightarrow \mathbb{R}$ is continuous at a point $w \in \mathbb{R}$ if 
$$
\lim_{v \rightarrow w} f(v) = f(w).
$$

A function $f$ is said to be continuous if it is continuous at all points
in its domain.

<br>

.center.width-80[![](./figures/fundamentals/continuity.png)]

---

## Differentiable functions

The derivative of $f: \mathbb{R} \rightarrow \mathbb{R}$ at $w \in \mathbb{R}$ is defined as
$$
f'(w) \coloneqq \lim_{\delta\rightarrow 0} \frac{f(w+\delta)- f(w)}{\delta}
$$
provided that the limit exists.
If $f'(w)$ is well-defined at a particular
$w$, we say that the function $f$ is differentiable at $w$.

<br>

.center.width-50[![](./figures/fundamentals/derivative.png)]

---

## Calculus rules

- Linearity: $\forall a, b \in \mathbb{R}, \ (af+bg)'(w) = af'(w) +bg'(w)$ 
- Product rule: $(fg)'(w) = f'(w) g(w) + f(w)g'(w)$
- Chain rule: $(f\circ g)'(w) = f'(g(w)) g'(w)$, where
$(f\circ g)(w) \coloneqq f(g(w))$

As the linearity and the product rules can be rederived from 
the chain rule, the chain rule can be seen as the cornerstone of differentiation.

---

## Leibniz notation

Suppose $u \coloneqq f(w)$

- Lagrange notation: $f'(w)$
- Leibniz notation: $f' = \frac{du}{dw}$

--

Now suppose $v \coloneqq g(w)$ and $u \coloneqq f(v)$

- Lagrange notation: $(f\circ g)'(w) = f'(g(w)) g'(w)$
- Leibniz notation: $\frac{du}{dw} = \frac{du}{dv} \cdot \frac{dv}{dw}$

---

## Directional derivatives

The directional derivative
of
$f \colon \mathbb{R}^P \to \mathbb{R}$ 
at 
$\bm{w} = (w\_1, ..., w\_P) \in \mathbb{R}^P$
in the direction $\bm{v} \in \mathbb{R}^P$ is given by 
$$
\partial f(\bm{w})[\bm{v}] \coloneqq \lim_{\delta \rightarrow 0} \frac{f(\bm{w} + \delta \bm{v}) - f(\bm{w})}{\delta},
$$
provided that the limit exists. 

<br>

.center.width-50[![](./figures/fundamentals/dir_deriv.png)]

Directional derivative of the curve 
$f:\mathbb{R} \rightarrow \mathbb{R}^2$ in direction $v=1$ is the tangent

---

## Partial derivatives

- The partial derivative for $i \in [P]$
$$
\partial\_i f(\bm{w}) 
\coloneqq \partial f(\bm{w})[\bm{e}\_i] 
= \lim_{\delta \rightarrow 0} \frac{f(\bm{w} + \delta \bm{e}\_i) - f(\bm{w})}{\delta}
$$
where
$$
\bm{e}_i \coloneqq (0, \ldots, 0, \underbrace{1}_i, 0, \ldots, 0).
$$

--

- Also denoted $\frac{\partial f(\bm{w})}{\partial w\_i}$ or $\partial\_{w\_i} f(\bm{w})$

--

- As if we are differentiating a function $\omega\_i \mapsto f(w\_1, \ldots,
\omega\_i, \ldots, w\_P)$ around $\omega\_i$, letting all other coordinates fixed
at their values $w\_i$.

---

## Gradients

- The gradient of a differentiable function $f: \mathbb{R}^P \rightarrow \mathbb{R}$
at a point $\bm{w} \in \mathbb{R}^P$ is defined as the vector of partial derivatives

$$
\nabla f(\bm{w}) 
\coloneqq \begin{pmatrix} \partial\_1 f(\bm{w}) \\\\
  \vdots \\\\
  \partial\_P f(\bm{w}) \end{pmatrix}
  = \begin{pmatrix} \partial f(\bm{w})[\bm{e}\_1] \\\\
  \vdots \\\\
  \partial f(\bm{w})[\bm{e}\_P] \end{pmatrix} \in \mathbb{R}^P
$$

.center.width-50[![](./figures/fundamentals/gradient.png)]

--

- Using $\bm{v}=\sum_{i=1}^P v\_i \bm{e}\_i$ and the linearity of the directional derivative:

$\partial f(\bm{w})[\bm{v}] 
= \sum_{i=1}^P v\_i \partial f(\bm{w})[\bm{e}\_i]
= \langle \bm{v}, \nabla f(\bm{w})\rangle$

---

## Why is the gradient useful?

---

## Jacobians

---

## Special cases of the Jacobian
