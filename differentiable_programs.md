class: middle, center, title-slide

$$
\gdef\x{\bm{x}}
\gdef\s{\bm{s}}
\gdef\w{\bm{w}}
\gdef\RR{\mathbb{R}}
\gdef\cS{\mathcal{S}}
\gdef\cW{\mathcal{W}}
$$

# Differentiable programming

Lecture 2: Differentiable programs

<br><br>
Mathieu Blondel, Vincent Roulet

---

# Outline

- Pararameterized programs
  * Computation chains
  * Feedforward networks
  * Multilayer perceptrons
  * Directed acyclic graphs (DAGs)
  * Computation graphs
- Control flows

---

## Computation chains

.center.width-100[![](./figures/differentiable_programs/chain.png)]

Sequence of functions
$$
\begin{aligned}
\s\_0 &\in \cS\_0\nonumber \\\\
\s\_1 &\coloneqq f\_1(\s\_0) \in \cS\_1 \\\\
     & \hspace{6pt} \vdots\\\\
\s\_K &\coloneqq f\_K(\s\_{K-1}) \in \cS\_K \\\\
f(\s\_0) &\coloneqq \s\_K
\end{aligned}
$$

Equivalent to function compositions
$$
\begin{aligned}
f(\s\_0)
&= (f\_K \circ \dots \circ f\_2 \circ f\_1)(\s\_0) \\\\
&= f\_K(\dots f\_2(f\_1(\s\_0))).
\end{aligned}
$$

---

## Feedforward networks = parameterized computain chains

.center.width-100[![](./figures/differentiable_programs/feedforward.png)]

$$
\begin{aligned}
\s\_0 &\coloneqq \x \\\\
\s\_1 &\coloneqq f\_1(\s\_0, \w\_1) \\\\
\s\_2 &\coloneqq f\_2(\s\_1, \w\_2) \\\\
&\vdots \\\\
\s\_K &\coloneqq f\_K(\s_{K-1}, \w\_K) \\\\
f(\x, \w) &\coloneqq \s\_K
\end{aligned}
$$

$$
\w \coloneqq (\w\_1, \dots, \w\_K) \in (\cW\_1, \dots, \cW\_K)
$$

---

## Multilayer perceptrons

<br>

$$
\begin{aligned}
\s\_0 &\coloneqq \x \\\\
\s\_1 &\coloneqq f\_1(\s\_0, \w\_1) \coloneqq a\_1(\bm{W}\_1 \s\_0 + \bm{b}\_1) \\\\
\s\_2 &\coloneqq f\_2(\s\_1, \w\_2) \coloneqq a\_2(\bm{W}\_2 \s\_1 + \bm{b}\_2) \\\\
&\vdots \\\\
\s\_K &\coloneqq f\_K(\s\_{K-1}, \w\_K) \coloneqq a\_K(\bm{W}\_K \s\_{K-1} + \bm{b}\_K)
\end{aligned}
$$
<br>
where 
* $\w\_k \coloneqq (\bm{W}\_k, \bm{b}\_k)$ are the parameters of layer $k$
* $a_k$ is an activation function (softplus, ReLu, ...)

<br>

We can see an MLP as a generalized linear model (GLM) <br>
$\s\_{K-1} \mapsto a\_K(\bm{W}\_K \s\_{K-1} + \bm{b}\_K)$
on top of a learned representation $\s_{K-1}$.

---

## Directed acyclic graphs (DAGs)

.center.width-50[![](./figures/differentiable_programs/graph.png)]

Nodes can have parent or children nodes.

DAGs define a topological order.

Examples: 
* (0,1,2,3,4)
* (0,2,1,3,4)

---

## Representing programs as DAGs

<br>

.center.width-100[![](./figures/differentiable_programs/graph_comput.png)]

<center>Representation of $f(x_1, x_2) \coloneqq x_2e^{x_1}\sqrt{x_1 + x_2 e^{x_1}}$ as a DAG</center>

<br>

Functions and variables are nodes.

Edges indicate function and variable dependencies.

The function $f$ is decomposed as $8$ elementary functions in topological order.

---

## Executing a program

<br>

.center.width-100[![](./figures/differentiable_programs/executing_program.png)]

---

## Representation used in the book

.center.width-70[![](./figures/differentiable_programs/graph1.png)]

Functions and output variables are represented by the same nodes.

<br>

## Alternative representation

.center.width-70[![](./figures/differentiable_programs/graph2.png)]

Functions and variables are represented by a disjoint set of nodes (bipartite graph).
