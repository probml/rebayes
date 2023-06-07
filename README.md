# ReBayes = Recursive Bayesian inference for latent states

📝 Paper: [Low-rank extended Kalman filtering for online learning of neural networks from streaming data](https://arxiv.org/abs/2305.19535)

We provide code for online (recursive) Bayesian inference in state space models;
in contrast to the dynamax code, we do not assume the entire observation sequence is available in advance,
so the ReBayes API can be used in an interactive loop (e.g., for Bayesian optimization).
We assume the dynamics model is linear Gaussian (or constant),
but the observation model can be non-linear and non-Gaussian.

This is work in progress; a stable version will be released late Spring 2023.

![flipbook-lofi-fourier](https://user-images.githubusercontent.com/4108759/230786889-9fabdada-20d4-49fc-b9ee-c67d4db90d4b.png)
