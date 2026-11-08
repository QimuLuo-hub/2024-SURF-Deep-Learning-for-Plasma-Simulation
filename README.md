# Deep Learning Method for Plasma Simulation

**An adaptive deep unfitted Nitsche method for elliptic interface problems**

XJTLU Summer Undergraduate Research Fellowship, project SURF-2024-0016 (Jun – Oct 2024)
Qimu Luo (team leader), Hanzhe Xue · Supervisors: Dr. Ren Zhao, Dr. Yiru Ye
Department of Foundational Mathematics, School of Mathematics and Physics

🏆 School Winner (School of Mathematics and Physics) · Excellent Poster Award · Student-Nominated Winner, SURF 2024

📄 [Poster](Poster-Adaptive.pdf) · [Slides](Slides.pdf) · [Report](Report.pdf)

## Problem

Elliptic interface problems arise in plasma and material simulations, where the diffusion coefficient jumps across an internal interface $\Gamma$:

$$
\begin{aligned}
-\nabla\cdot\big(\beta(x)\nabla u(x)\big) &= f && \text{in } \Omega_1 \cup \Omega_2, \\
u &= g && \text{on } \partial\Omega, \\
[\![u]\!] = p, \quad [\![\beta\,\partial_n u]\!] &= q && \text{on } \Gamma,
\end{aligned}
$$

where $\beta = \beta_i$ is constant on each subdomain $\Omega_i$ and $[\![\cdot]\!]$ denotes the jump across $\Gamma$.

## Method

1. **Deep unfitted Nitsche method.** Two weakly coupled neural networks $u_{\theta_1}$ and $u_{\theta_2}$ approximate the solution on either side of the interface. They are trained by minimising a Monte-Carlo discretisation of the **Nitsche energy functional**, which imposes the jump conditions weakly through weighted averages $\{\!\{\cdot\}\!\}$ and a penalty $\gamma_f$. The boundary condition is enforced with a penalty $\gamma_b$. No interface-fitted mesh is needed.
2. **Gradient-based adaptive sampling.** Training proceeds in refinement levels:
   - train the networks for a fixed number of epochs;
   - compute $|\nabla u_\theta|$ at the interior, interface and boundary points;
   - keep the top-ranked 10% of points and refine locally around them (new grid points near the selected points, extra points on the interface);
   - repeat with the enlarged training set.

   This places training points where the solution changes quickly, near the interface.

The [report](Report.pdf) also covers the derivation (Nitsche's method for boundary and interface conditions, energy minimisation, residual networks, Monte-Carlo quadrature). It tests the non-adaptive method on 2D interfaces (straight, high-contrast, flower, sunflower), 3D interfaces (ellipsoid, popcorn) and higher-dimensional examples.

## Results

The `results/` spreadsheets record the $L^2$ error and the number of training points at each refinement level:

| Interface | Setting | Level 1 error | Level 6 error | Points $(x_r, x_f, x_b)$ at level 6 |
|---|---|---|---|---|
| Flower | $\gamma_b = 3000$, $\gamma_f = 150$ | 0.138 | 0.027 | (402, 121, 52) |
| Sunflower, 15 petals ($\beta_1 = 1$, $\beta_2 = 10$) | $\gamma_b = 8000$, $\gamma_f = 250$ | 1.309 | 0.035 | (452, 153, 52) |

Sunflower setup (from the poster): domain $[-1,1]^2$, interface $r(\theta) = 0.4 + 0.2\sin(15\theta)$, initial $10\times 10$ grid with $N_f = 75$ interface points, 64 neurons × 5 layers, tanh, Adam ($\text{lr}=10^{-3}$), 1500 epochs per level, 6 refinement levels.

## Repository structure

```
├── Poster-Adaptive.pdf          # SURF 2024 poster
├── Slides.pdf                   # SURF symposium presentation (Oct 2024)
├── Report.pdf                   # full technical report (46 pp.)
├── code/
│   ├── adaptive-2d/
│   │   ├── indicator.py         # residual- and gradient-based error indicators
│   │   ├── flower_refine.py     # adaptive training, flower-shaped interface
│   │   ├── sunflower_refine.py  # adaptive training, sunflower interface
│   │   └── visualize_adapt.py   # solution and error surface plots
│   └── basesline/               # (placeholder)
└── results/
    ├── flower/                  # per-level sampling plots, error/loss curves, .xlsx tables
    └── sunflower/
```

**Note:** the adaptive scripts import the base deep-Nitsche solver (`DUNM.Module.*`: networks, Nitsche loss, interface definitions), which is not included in this repository. A self-contained, later version of the 2D and 3D code is in [Gradient-Based-Adaptive-Elliptic-Interface-Problem](https://github.com/QimuLuo-hub/Gradient-Based-Adaptive-Elliptic-Interface-Problem).

## Follow-up

The project was extended to an **energy-based** adaptive indicator and published as:

> Qimu Luo, Ren Zhao. Energy-Based Adaptive Deep Unfitted Nitsche Method for Elliptic Interface Problems. *East Asian Journal on Applied Mathematics*, 16(3): 573–599, 2026. [DOI: 10.4208/eajam.2025-069.100925](https://doi.org/10.4208/eajam.2025-069.100925) · [Code](https://github.com/QimuLuo-hub/Energy-Based-Adaptive-Deep-Unfitted-Nitsche-Method-for-Elliptic-Interface-Problems)
