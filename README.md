# Physics-Driven Deep Learning for PDEs and Inverse Problems
This repository implements advanced **physics-driven deep learning frameworks** for solving partial differential equations (PDEs) and related inverse problems. These methods leverage deep neural networks to model PDE solutions efficiently while incorporating physical constraints.

## 📌 Framework Categories

The implemented methods can be broadly categorized into three types:
#### 1️⃣ Strong-Form-Based Approaches
These methods are based on the strong form of PDEs, meaning they directly enforce the differential equations as constraints:
- **Advantages**: Simple to implement, does not require integral calculations.
- **Disadvantages**: Requires high-order derivatives and imposes higher regularity requirements on the solution.

Related methods (implemented methods are bolded):
- **[Physics-Informed Neural Networks (PINNs)](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)**

#### 2️⃣ Weak-Form-Based Approaches
These methods use the weak form of PDEs, making them more flexible in handling lower regularity solutions:
- Advantages: Avoids the need for high-order derivatives, accommodates lower-regularity solutions.
- Disadvantages: Requires integral calculations, often leading to a large number of integration points; some frameworks define test functions globally, making them less effective for complex geometries.

Related methods (implemented methods are bolded):
- [Weak Adversarial Networks (WANs)](https://www.sciencedirect.com/science/article/abs/pii/S0021999120301832)
- [Deep Ritz Method (DeepRitz)](https://link.springer.com/article/10.1007/s40304-018-0127-z): Solves PDEs by reformulating them as an equivalent variational problem, which requires the PDE to have an energy functional.
- [Variational Physics-Informed Neural Networks (VPINNs)](https://arxiv.org/abs/1912.00873)
- **[ParticleWNN](https://arxiv.org/abs/2305.12433)**: Uses Compactly Supported Radial Basis Functions (CSRBFs) as test functions, reducing the need for many integration points and making it suitable for complex geometries.


#### 3️⃣ Hybrid & Other Approaches
These frameworks combine deep neural networks with traditional numerical methods, improving efficiency and accuracy while still inheriting some limitations of classical methods (e.g., curse of dimensionality).

## 🏗️ Implemented PDE Examples
We provide implementations of various PDEs to demonstrate the effectiveness of different methods:
- Poisson’s Equation
- Darcy’s Flow
- Linear Elasticity Equation
- Stokes Flow
- Steady Navier-Stokes (NS) Equation
- Burgers’ Equation
- Allen-Cahn Equation
- Wave Equation

## 🚀 Future Work
We will continue expanding this repository by adding more frameworks, solving additional PDEs, and incorporating inverse problems. Contributions and discussions are welcome!