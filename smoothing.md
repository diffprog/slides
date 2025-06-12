class: middle, center, title-slide

$$
\gdef\u{\bm{u}}
\gdef\v{\bm{v}}
\gdef\z{\bm{z}}
\gdef\muv{\bm{\mu}}
\gdef\RR{\mathbb{R}}
$$

# The Elements of <br> Differentiable Programming

**Part IV: Smoothing programs**

<br><br>
Mathieu Blondel, Vincent Roulet

---

name: optim

.center.smaller[**Part IV: Smoothing programs**]

- **Smoothing by optimization**
  * Infimal convolution
  * Moreau envelope
  * Dual approach
- <a class="outline-link" href="#integration">Smoothing by integration</a>

---

## Infimal convolution

The infimal convolution between $f$ and $g$ is a new function $f \square g$

$$
\begin{aligned}
(f \square g)(\muv) 
&\coloneqq \inf\_{\u \in \RR^M} f(\u) + g(\muv - \u) \\\\
&= \inf\_{\z \in \RR^M} f(\muv + \z) + g(\z) \\\\
&= \inf\_{\u, \z \in \RR^M} f(\u) + g(\z) \text{ s.t. } \u = \muv + \z.
\end{aligned}
$$
The change of variable $\u \coloneqq \muv + \z$ is the location-scale transform we saw before.

<br>

**Link with classical convolution**

Counterpart of convolution where **integration** is replaced with **minimization**.

<br>

**Commutativity**

$$
(f \square g)(\muv) = (g \square f)(\muv) \quad \forall \muv \in \RR^M
$$

---

## Moreau envelope

The Moreau envelope of a function $f$ is a **smooth approximation** of it.

It is equal to the infimum colution of $f$ with $R(\z) \coloneqq \frac{1}{2} \\|\z\\|^2$

$$
\begin{aligned}
\mathrm{env}\_f(\muv) 
&\coloneqq (f \square R)(\muv) \\\\
&= \inf\_{\u \in \RR^M} f(\u) + R(\muv - \u) \\\\
&= \inf\_{\z \in \RR^M} f(\muv + \z) + R(\z) \\\\
&= \inf\_{\u, \z \in \RR^M} f(\u) + R(\z) \text{ s.t. } \u = \muv + \z.
\end{aligned}
$$

Compare with the proximal operator of $f$
$$
\mathrm{prox}\_{f}(\muv) 
\coloneqq \argmin\_{\u \in \RR^M} f(\u) + R(\muv - \u) 
$$

---

## Example: Huber loss

The Moreau envelope of $f(\u) \coloneqq \\|\u\\|\_1 = \sum\_{j=1}^M |u\_j|$ is the Huber loss
$$
\mathrm{env}\_f(\muv) = \sum\_{j=1}^M \mathrm{huber}(\mu\_j) \approx \sum_{j=1}^M |\mu\_j|
$$
where
$$
\mathrm{huber}(\mu\_j)
\coloneqq
\begin{cases}
\frac{\mu\_j^2}{2} &\text{ if } |\mu\_j| \le 1 \\\\
|\mu\_j| - \frac{1}{2} &\text{ if } |\mu\_j| > 1
\end{cases}
$$

.center.width-55[![](./figures/smoothing/huber_abs.png)]

---

class: middle

.center.width-100[![](./figures/smoothing/moreau_env_non_cvx.png)]

---

## Legendre-Fenchel transform (convex conjugate)

The Legendre-Fenchel transform of $f$ is another function denoted $f^\*$
$$
f^\*(\v) \coloneqq \sup\_{\u \in \mathrm{dom}(f)}
\langle \u, \v \rangle - f(\u)
$$

<br>

.center.width-80[![](./figures/smoothing/tightest_lower_bound.png)]
$u \mapsto uv - f^\*(v)$
is the tighest affine lower bound of $f$ with a fixed slope $v$.

---

class: middle

.center.width-100[![](./figures/smoothing/conjugate.png)]

Instead of representing a convex function $f$ by its graph $(\u, f(\u))$
we can represent it by the set of tangents with slope $\v$ and
intercept $-f^\*(\v)$.

---

## Dual approach

If $f$ and $R$ are both convex and closed, we have
$$
f \square R = (f^\* + R^\*)^\*
$$
Go to the dual $f^\*$, add regularization $R^\*$ and come back to the primal.

The dual approach is often more convenient.

**Example: smoothed ReLU**

* $f(u) = \max(u, 0)$
* $f^\*(v) = \iota_{[0,1]}(v) = 0 \text{ if } v \in [0,1] \text{ else } \infty$
* $(f + R^\*)^\*(u) \approx f(u)$

.center.width-50[![](./figures/smoothing/smoothed_relu.png)]

---

name: integration

.center.smaller[**Part IV: Smoothing programs**]

- <a class="outline-link" href="#optim">Smoothing by optimization</a>
- **Smoothing by integration**
  * Convolution
  * Perturbation of blackbox functions
  * Gumbel tricks
