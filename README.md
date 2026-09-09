# Safe Autonomous Driving : Comparison of MPC-CBF and RL controllers

**Author:** Santiago Parra  
**Year:** 2026

This project develops a MPC controller combined with a Control Barrier Function (CBF), and compares it with a RL based controller for autonomous driving with Metadrive.

<p align="center">
 <img src="src/media/mpc_cbf_demo.gif" width="600" alt="MPC-CBF Demo">
</p>

## MPC-CBF Controller : Formulation of the problem

The MPC controller is a classic controller an intuitive perspective for control driving. Given a time horizon T, the idea is to solve the following optimization problem for every time step of the simulation:

$\min J(u) = \sum_{k=1}^N \omega_P J_{pos}(x_k) + \omega_S J_{speed}(x_k) + \omega_U J_{control}(u_k)$

Where $x_k$ represents the state of the vehicle and $u_k$ represents the sequence of actions / controls that guide the vehicle. The weights $\omega_P$, $\omega_S$ and $\omega_U$ penalize the position, the speed and the control respectively.

This cost function represents the desired behaviour of the vehicle (go straight in the lane, go at a reasonable speed, don't do sudden moves, etc).
The optimization should be subject to the following constraints:

- The dynamics of the system: Represents the reaction of the vehicle when we choose an action $$x_{k+1} = f(x_k, u_k)$$
- The Control Barrier Function (CBF): We create a barrier around each obstacle that the vehicle cannot cross, we represent this behaviour by a positive barrier function $h(x) \geq 0$. In order to guarantee the positivity, when solving the optimization problem we impose:
  $$h(x_{k+1}) \ge (1 - \gamma) h(x_k)$$
- Limits of the control: As the control represents a real life action, it should be bounded (for example, the steer cannot exceed 45° or the acceleration has a maximum limit). We represent this by a normalized constraint for the controls
  $$-1 \leq u_k \leq 1 \hspace{10pt}$$

Which are taken for each $k = 1, 2, \dots N$ i.e for each instant of the time horizon.

In our implementation, we perform a LTV approximation in order to transform the optimization problem into an Quadratic Programming (QP) problem. In the end we solve a system of the form:

$$\min \frac{1}{2} U^T P U + q^T U$$ 

Subject to a linear constraint $l \leq A U \leq u$ that encapsulates the CBF constraint and the physical limit of the control.

## RL Controller

- In addition to the MPC-CBF, we trained a **Proximal Policy Optimization (PPO)** controller using the `stable-baselines3` package with the Metadrive environment. 
- We trained the model using a MLP Policy over 50 generated scenarios with traffic density of 0.15 over 250000 timesteps. 
- The MLP receives the Metadrive observations (state of the vehicle + LiDAR measurements) and returns the chosen control (steer, acceleration) for the case.
- The cost function for the learning process is the standard PPO cost function. We provide to PPO the  default reward function coming in `MetaDriveEnv`, which is defined as:

$$ R = c_1 R_{driving} + c_2 R_{speed} + R_{terminate}$$

Where $R_{driving}$ and $R_{speed}$ motivate the desired behaviour of position and speed of the vehicle respect to reference target values, respectively (similar to $J_{pos}$, $J_{speed}$ in the MPC-CBF formulation). $R_{terminate}$ contains a set of rewards: the success of the episode, penalization to crashes, etc.

We consider here the default weight configuration ($c_1$, $c_2$, etc) which can be consulted in the Metadrive docs.

**Note:** Here the crash constraint is given as a penalty on the reward function. This is called a soft-formulation, compared to the explicit constraint formulation given in the MPC-CBF.

Additionally for the RL controller, we added a Safety Filter, in which we impose a progressive deacceleration if the vehicle reaches certain speed limit and an Emergency Braking System when approaching nearby vehicles

## Experiments

For the experiments, we considered a $N = 15$, $\Delta t = 0.1$, a $\gamma = 0.2$ and the Kinematic Bicycle model, where the control is given by the acceleration and steer of the vehicle. We will compute over all the experiments:
- **Avg Lateral Error:** The mean deviation of the vehicle from the reference lane 
- **Max Lateral Dev:** The max deviation of the vehicle from the reference lane
- **Min Safety Dist:** The minimum distance of safety held with respect the other vehicles
- **Avg Jerk:** Measures the average change of acceleration of the vehicle
- **Steer Rate:** Measures the average change in the steer (the average speed when turning the wheel)

We tested both controllers in 10 different scenarios with a single map of the type "CCCC", and we got the following metrics

| Controller | Success Rate | Avg Lateral Error (m) | Max Lateral Deviation (m) | Min Safety Dist (m) | Avg Jerk | Steer Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MPC-CBF** | 50% | **0.30 ± 0.06** | **0.34 ± 0.43** | **3.33 ± 0.51** | **0.51 ± 0.19** | **0.23 ± 0.07** |
| **RL** | 50% | 0.96 ± 0.23 | 1.25 ± 1.78 | 5.23 ± 2.77 | 7.83 ± 1.06 | 0.45 ± 0.04|

We see that the MPC-CBF controller overcomes the RL controller in every metric. The lateral deviation metrics show that the MPC-CBF gets higher accuracy following the lane, at the same time that it gets a higher comfort (less jerk and steer rate) in comparison to the RL scheme, which seems to be more violent and aggressive with its driving. We can also see that the MPC-CBF is "reactive" as it approaches closer to the other vehicles, but always keeps an almost deterministic behaviour. In contrast, the RL model is more impredictable due to its high variance in the safety distance.

### Comparison with different time horizons

We can also try different horizons for the MPC-CBF controller. Specifically, we will try the values $N=5$, $N=15$ and $N=30$, and compute the same metrics. We get the following results:

| Horizon (N) | Success Rate | Avg Lateral Error (m) | Max Lateral Deviation (m) | Min Safety Dist (m) | Avg Jerk | Steer Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $N = 5$ | 0.0 % | 3.60 ± 2.68 | 3.01 ± 2.05| 19.08 ± 19.64 | 0.10 ± 0.11 | 0.02 ± 0.03|
| $N = 15$ | **50 %** | **0.30 ± 0.06** | **0.34 ± 0.43** | **3.33 ± 0.51** | **0.51 ± 0.19** | **0.23 ± 0.07** |
| $N = 30$ | 10 % | 0.34 ± 0.07 | 0.30 ± 0.35 | 3.17 ± 0.46 | 0.53 ± 0.16 | 0.36 ± 0.12 |

We see different behaviours depending on the size of the horizon: 
For a low horizon $N=5$, the vehicle becomes blind to the future and hence its responses becomes very reactive. That's why we get higher lateral deviations with low response rates (low jerk and steer). In other words, the vehicle doesn't have the capacity to follow the lane before even reaching traffic.

When we turn to a horizon $N=15$ we get better metrics in comparison to the previous case: the vehicle starts following the lane as it anticipates more to the future, which is seen by the low lateral deviation metrics. We also get smoother behaviour and the vehicle avoids the obstacles without being too conservative (we can adjust this with the $\gamma$ parameter of the CBF).
The success rate increases considerably, suggesting the vehicle arrives to its destination.

The greater horizon with $N=30$ gets similar metrics, but as we use a LTV approximation to turn the MPC into a QP problem the linear approximation of the dynamics starts diverging, which translates into a bad performance when following of the lanes and avoiding obstacles. That's why, even when we get a similar behaviour to $N=15$, the success rate collapses, because the model starts hallucinating the real dynamics of the vehicle.

### Comparison of CBF Implementation:

Now we will compare the impact of changing the CBF implementation. In the original formulation, the CBF is implemented as an explicit constraint on the MPC problem, which must hold along the entire time horizon. Another alternative could be to implement an unconstrained MPC problem and then implement a **Safety Filter**. Mathematically, in the time instant $k$, we would solve the QP problem

$$\min_u \|u - u_{MPC}\|^2$$
Subject to the CBF constraint (which can be then linearized as before) $$h(x_{k+1}) \ge (1 - \gamma) h(x_k)$$

(This means: to project the original MPC solution into the safety set in the actual instant)
Note that the CBF constraint is explicitly taken for the actual time instant, which contrasts to the original formulation, which ensures that the CBF constraint is hold for the entire time horizon. This translates to a faster computation of the control, but sacrifices the strong safety constraint.

Just like the previous experiments, we take 10 random scenarios, and compare the performance of both methods.

| Controller |  Success Rate | Avg Lateral Error (m) | Max Lateral Deviation (m) | Min Safety Dist (m) | Avg Jerk | Steer Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Joint MPC-CBF** | **50%** | **0.30 ± 0.06** | **0.34 ± 0.43** | **3.33 ± 0.51** | **0.51 ± 0.19** | **0.23 ± 0.07** |
| **MPC + CBF (Safety Filter)** | 10% | 0.24 ± 0.05 | 0.17 ± 0.13 | 3.51 ± 0.71 | 0.55 ± 0.25 | 0.21 ± 0.13 |

We see that the MPC-CBF strategy continues to outcome the Safety Filter method when it comes to taking the vehicle to the destination. However, both methods are similar when it comes to comfort and safety distance from the other vehicles. Moreover, the Safety Filter gets better metrics when it comes to follow the lane. We expect this, since the safety filter is a projection of the MPC solution which is designed precisely to follow the lane.