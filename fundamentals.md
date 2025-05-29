class: middle, center, title-slide

$$
\gdef\x{\bm{x}}
\gdef\w{\bm{w}}
\gdef\v{\bm{v}}
\gdef\RR{\mathbb{R}}
\gdef\jac{\bm{\partial}}
\gdef\cE{\mathcal{E}}
\gdef\cF{\mathcal{F}}
$$

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

*Directional derivative of the curve 
$f:\mathbb{R} \rightarrow \mathbb{R}^2$ in direction $v=1$ is the tangent*

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

The gradient of a differentiable function $f: \mathbb{R}^P \rightarrow \mathbb{R}$
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

--

<br><br>

Using $\bm{v}=\sum_{i=1}^P v\_i \bm{e}\_i$ and the linearity of the directional derivative:

$\partial f(\bm{w})[\bm{v}] 
= \sum_{i=1}^P v\_i \partial f(\bm{w})[\bm{e}\_i]
= \langle \bm{v}, \nabla f(\bm{w})\rangle$

---

## Why is the gradient useful?

We say that $\bm{v}$ is an **ascent direction** of $f$ from $\bm{w}$ if
$$
\langle \bm{v}, \nabla f(\bm{w}) \rangle > 0.
$$

--

We can then seek the **steepest ascent direction**
$$
\argmax\_{\bm{v} \in \mathbb{R}^P, \\|\bm{v}\\|\_2 \leq 1}
\langle \bm{v}, \nabla f(\bm{w}) \rangle
= \argmax\_{\bm{v} \in \mathbb{R}^P, \\|\bm{v}\\|\_2 \le 1}
\partial f(\bm{w})[\bm{v}]
= \frac{\nabla f(\bm{w})}{\\|\nabla f(\bm{w})\\|\_2}
$$

<br>

.center.width-50[![](./figures/fundamentals/gradient.png)]

---

## Jacobians

The Jacobian of a differentiable function $f: \mathbb{R}^P \rightarrow \mathbb{R}^M$ at $\bm{w}$ is
defined as the matrix gathering partial derivatives of each coordinate's function

$$
\bm{\partial} f (\bm{w}) 
\coloneqq \begin{pmatrix}
        \partial\_1 f\_1(\bm{w}) & \ldots & \partial\_P f\_1(\bm{w}) \\\\
        \vdots & \ddots & \vdots \\\\
        \partial\_1 f\_M(\bm{w}) & \ldots & \partial\_P f\_M(\bm{w})
    \end{pmatrix} \in \mathbb{R}^{M \times P}
$$

<br>

--

The Jacobian can be represented by stacking columns of partial derivatives or rows
of gradients,
$$
\bm{\partial} f (\w) 
= \begin{pmatrix}
        \partial\_1 f(\w), \ldots, \partial\_P f(\w)
    \end{pmatrix} 
  = \begin{pmatrix}
    \nabla f\_1(\w)^\top \\\\
    \vdots \\\\
    \nabla f\_M(\w)^\top
    \end{pmatrix} \in \mathbb{R}^{M \times P}
$$

--

<br>

Careful: if $f: \mathbb{R}^P \rightarrow \mathbb{R}$, then
$\jac f(\w) = \nabla f(\w)^\top \in \RR^{1\times P}$

---

## Example: Jacobian of an element-wise activation function σ

Suppose $f \colon \RR^P \to \RR^P$ is defined as
$$
f(\w) 
\coloneqq \begin{pmatrix}
    \sigma(w\_1) \\\\
    \vdots \\\\
    \sigma(w\_P)
\end{pmatrix}\in \RR^P
$$
Then
$$
\bm{\partial} f(\w) 
= \mathrm{diag}(\sigma'(w\_1), \dots, \sigma'(w\_P))
\coloneqq \begin{pmatrix}
    \sigma'(w\_1) & 0 & \ldots & 0 \\\\
    0 & \ddots & \ddots & \vdots \\\\
    \vdots & \ddots & \ddots & 0 \\\\
    0 & \ldots & 0 & \sigma'(w\_P)
\end{pmatrix} \in \RR^{P \times P}
$$

---

## Chain rule

<br>

Consider $f:\RR^P \rightarrow\RR^M$ and $g:\RR^M \rightarrow \RR^R$. Then,
$$
\underbrace{\jac (g\circ f)(\w)}\_{\RR^{R \times P}} = \underbrace{\jac g(f(\w))}\_{\RR^{R \times M}} \underbrace{\jac f(\w)}\_{\RR^{M \times P}}
$$

*Example:* $f$ a layer, $g$ another layer

<br><br>

--


Consider $f:\RR^P \rightarrow\RR^M$ and $L:\RR^M \rightarrow \RR$. Then,
$$
\nabla (L \circ f)(\w) 
= \jac L(f(\w)) \jac f(\w)
= \jac f(\w)^\top \nabla L(f(\w))
$$

*Example:* $L$: loss function, $f$: neural network


---

## The need for linear maps

Suppose we want to differentiate a function $f \colon \RR^{M \times D} \to M$ defined by
$$f(\bm{W}) \coloneqq \bm{W} \x$$
where $\bm{W} \in \RR^{M \times D}$ and $\x \in \RR^D$

--

We could always differentiate $\tilde{f}(\w)$ where $\w \coloneqq \mathrm{vec}(\bm{W}) \in \RR^{MD}$

--

However, the Jacobian $\jac \tilde{f}(\w) \in \RR^{M \times MD}$ can be shown to be extremely **sparse**

--

Fortunately, we never to materialize the Jacobian as a matrix.

We can directly see $\partial f(\bm{W})$ as a **linear map** (a.k.a. **linear operator**)

---

## Jacobian-vector products
