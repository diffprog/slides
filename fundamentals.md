class: middle, center, title-slide

$$
\gdef\e{\bm{e}}
\gdef\u{\bm{u}}
\gdef\v{\bm{v}}
\gdef\x{\bm{x}}
\gdef\w{\bm{w}}
\gdef\RR{\mathbb{R}}
\gdef\jac{\bm{\partial}}
\gdef\cE{\mathcal{E}}
\gdef\cF{\mathcal{F}}
\gdef\cG{\mathcal{G}}
$$

# The Elements of <br> Differentiable Programming

**Part I: Fundamentals**

<br><br>
Mathieu Blondel, Vincent Roulet

---

name: diff

.center.smaller[**Part I: Fundamentals**]

- **Differentiation**
  * Continuous functions
  * Differentiable functions
  * Gradients
  * Jacobians
  * Linear maps: JVPs and VJPs
  * Hessians and Hessian-vector products (HVPs)
- <a class="outline-link" href="#probaLearning">Probabilistic learning</a>

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

.center.width-100[![](./figures/fundamentals/continuity.png)]

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

<br>

As the linearity and the product rules can be rederived from 
the chain rule, the chain rule can be seen as the cornerstone of differentiation.

---

## Leibniz notation

Suppose $u \coloneqq f(w)$

- Lagrange notation: $f'(w)$
- Leibniz notation: $f' = \frac{du}{dw}$

--

<br>

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

--

--- 

<br>

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

## Jacobian-vector products (JVPs)

Suppose $f \colon \cE \to \cF$,
where $\cE$ and $\cF$ are general Euclidean spaces.

We can see the directional derivative
$$
\v \mapsto \partial f(\w)[\v]
$$
as a linear map $\cE \to \cF$

<br>

--

Therefore $\partial f \colon \cE \to (\cE \to \cF)$

--

<br>

We rarely need to materialize the Jacobian $\jac f(\w)$ as a matrix.

---

## Example of JVP

Let us go back to the example $f(\w) \coloneqq (\sigma(w\_1), \dots, \sigma(w\_P))$
with Jacobian
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

<br>

--

We can compute the JVP by element-wise multiplication
$$
\partial f(\w)[\v] = (\sigma'(w\_1), \dots, \sigma'(w\_P)) \circ \v
$$

<br>

--

Computational cost is $O(P)$ instead of $O(P^2)$ had we used a matrix-vector multiplication.

---

## Variations along outputs

Consider a function $f \colon \RR^P \to \RR^M$

Directional derivative and JVP: variations of $f$ along an **input** direction $\v \in \RR^P$

--

Instead, we may consider variations along an **output** direction $\u \in \RR^M$

$$
\nabla \langle \u, f \rangle(\w) = \jac f(\w)^\top \u
$$

<br>

where

$$
\langle \u, f \rangle(\w) \coloneqq \langle \u, f(\w) \rangle \in \RR.
$$

<br>
Using the concept of adjoint, this leads to the vector-Jacobian product.

---

## Adjoint maps

The adjoint of a linear map
$$
l \colon \cE \to \cF
$$
is another linear map
$$
l^\* \colon \cF \to \cE
$$
and satisfies
$$
\langle l[\v], \u \rangle = \langle \v, l^\*[\u] \rangle
$$
<br>
The adjoint is the counterpart of transpose for linear maps
$$
\langle \bm{A} \v, \u \rangle = \langle \v, \bm{A}^\top \u \rangle
$$

---

## Vector-Jacobian products (VJPs)

Suppose $f \colon \cE \to \cF$,
where $\cE$ and $\cF$ are general Euclidean spaces.

We can see
$$
\u \mapsto \partial f(\w)^\*[\u] = \nabla \langle \u, f \rangle(\w)
$$
as a linear map $\cF \to \cE$

<br>

--

Therefore $\partial f(\cdot)^\* \colon \cE \to (\cF \to \cE)$

---

class: middle

.center[Jacobian-vector product (JVP) $\v \mapsto \partial f(\w)[\v]$]

.center.width-60[![](./figures/fundamentals/jvp_vjp_recap.png)]

.center[vector-Jacobian product (VJP) $\textcolor{chocolate}{\u \mapsto \partial f(\w)^*[\u]}$]

---

## Chain rule using linear maps

Consider $f:\cE \rightarrow \cF$ and $g:\cF \rightarrow \cG$, where $\cE$,
$\cF$ and $\cG$ are Euclidean spaces.

<br>

.center[
$
\partial (g\circ f)(\w)[\v] 
= \partial g(f(\w))[\partial f(\w)[\v]]
~ \forall \v \in \cE
$]

.center.width-80[![](./figures/fundamentals/chain_rule_recap.png)]

.center[
$
\textcolor{chocolate}{\partial (g\circ f)(\w)^\*[\u] 
= \partial f(\w)^\*[\partial g(f(\w))^\*[\u]]
~ \forall \u \in \cG}
$]

<br>

These two formulas are the basis of forward-mode and reverse-mode autodiff!

---

## Second derivatives

The derivative $f'$ is itself a function so we may want to differentiate it.

The seecond derivative $f^{(2)}(w)$ of a differentiable function $f:
\RR \rightarrow \RR$ at $w \in \RR$ is defined as the derivative of $f'$ at
$w$
$$
f^{(2)}(w) \coloneqq f''(w)
\coloneqq \lim_{\delta \rightarrow 0} \frac{f'(w+\delta) - f'(w)}{\delta}
$$
provided that the limit exists

.center.width-60[![](./figures/fundamentals/second_der.png)]

---

## Second directional derivatives

The second directional derivative of $f:\RR^P \rightarrow \RR$ at
$\w\in \RR^P$ along $\v, \v' \in \RR^P$ is defined as the directional
derivative of $\w \mapsto \partial f(\w)[\v]$ along $\v'$,

<br>
$$
\partial^2 f(\w)[\v, \v'] 
\coloneqq \lim_{\delta \rightarrow 0} 
\frac{\partial f(\w + \delta \v')[\v] - \partial f(\w)[\v]}{\delta},
$$

<br>

provided that $\partial f(\w)[\v]$ is well-defined around $\w$ and that the
limit exists.

---

## Second partial derivatives

The second partial derivatives are defined

<br>

$$
\partial_{i j}^2 f(\w) \coloneqq \partial^2 f(\w)[\e\_i, \e\_j] 
$$

<br>

where $\e\_i$, $\e\_j$ the $i$-th and $j$-th canonical directions in $\RR^P$

---

## Hessians

The Hessian of a twice differentiable function $f:\RR^P \rightarrow
\RR$ at $\w$ is the $P \times P$ matrix gathering all second partial
derivatives,

<br>

$$
\nabla^2 f(\w) \coloneqq \begin{pmatrix}
  \partial\_{11} f(\w) & \ldots & \partial\_{1P} f(\w) \\\\
  \vdots & \ddots & \vdots \\\\
  \partial\_{P1} f(\w) & \ldots & \partial\_{PP} f(\w)
\end{pmatrix} \in \RR^{P \times P}
$$

<br>

Note that for all $\v, \v' \in \RR^P$,
$$
\partial^2 f(\w)[\v, \v'] = \langle \v, \nabla^2 f(\w) \v'\rangle
$$

---

## Hessian-vector products

Oftentimes, we don't need the whole Hessian but only need to multiply with it.

Consider the function $f \colon \cE \to \RR$.

Then for any $\w, \v \in \cE$, the linear map
$$
\v \mapsto \nabla^2 f(\w)[\v]
$$ 
is called the Hessian-vector product (HVP).

---

name: probaLearning

.center.smaller[**Part I: Fundamentals**]

- <a class="outline-link" href="#diff">Differentiation</a>
  * Continuous functions
  * Differentiable functions
  * Gradients
  * Jacobians
  * Linear maps: JVPs and VJPs
  * Hessians and Hessian-vector products (HVPs)
- **Probabilistic learning**
