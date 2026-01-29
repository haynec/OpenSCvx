# What is Succesive Convexification?
!!! Warning
    This page is still under development :construction:.

*Successive Convexification* is an approach to solve infinite dimensional nonconvex trajectory optimization problems. It works by *successively* convexifying or linearizing a problem and solving the convex subproblem. The solution to the convex subproblem is then used to update the original problem, and the process is repeated until convergence.

![CT-SCvx_dark]{ align=left }
![CT-SCvx_light]{ align=left }


## Problem Formulation
In this repository, the user will likely find it most useful specify there problem in the Mayer Form order to take full advantage of the features of this repo, but not worry this is quite easy.

$$
\begin{align}
\min_{x,u}\ &L_{f}(x(t_f)), \\
\mathrm{s.t.}\ &\dot{x}(t) = f(t, x(t),u(t)) & \forall t\in[t_i, t_f], \\
& g(t,x(t),u(t)) \leq 0_{n_g} & \forall t\in[t_i, t_f],\\
& h(t,x(t),u(t)) = 0_{n_h} & \forall t\in[t_i, t_f],\\
& P(t_i, x(t_i), t_f, x(t_f)) = 0_{n_P} , \\
& Q(t_i, x(t_i), t_f, x(t_f)) = 0_{n_Q} , \\
\end{align}
$$

Lets break down whats happening here. The first line, $L_{f}(x(t_f))$, is specifying a terminal cost as a function of state. The second is describing the nonlinear dynamics of the system, $\dot{x}(t) = f(t, x(t),u(t))$ where $x$ and $u$ are the system state and control respectively. The third and fourth lines are describing the inequality, $g(t,x(t),u(t)) \leq 0_{n_g}$,  and equality, $h(t,x(t),u(t)) = 0_{n_h}$, constraints on the system respectively. Finally, the initial and terminal inequality and equality constraints are specified by $P(t_i, x(t_i), t_f, x(t_f)) = 0_{n_P}$ and $Q(t_i, x(t_i), t_f, x(t_f)) = 0_{n_Q}$ respectively.

<!-- 
## Simple Example
At face value, this may appear to be limiting, for example minimal fuel problems traditionally have a cost of the form $\sum\|u\|_2$. Lets consider a simple example of a double integrator with minimal fuel cost. 
### Dynamics
The discrete dynamics will be given by:

$$
\begin{gather}
\dot{r}(t) = r(t) + dt \cdot v(t), \\
\dot{v}(t) = v(t) + dt \cdot u(t).
\end{gather}
$$

where $r\in\mathbb{R}^2$ and $v\in\mathbb{R}^2$ are the position and velocity of the system respectively. The control which is accleration is given by $u\in\mathbb{R}^2$. The time step is given by $dt$. 

### Cost
We want a minimum fuel cost which will look something like the following,

$$\int^{t_f}_{t_i} \|u(t)\|_2\, dt$$ -->


[CT-SCvx_dark]: ../assets/images/ct-scvx_dark.png#only-dark
[CT-SCvx_light]: ../assets/images/ct-scvx_light.png#only-light