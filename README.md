# tGPLVM
## A Nonparametric, Generative Model for Manifold Learning with scRNA-seq experimental data

Input: A csv or hdf5 of scRNA count (or other data) with format *N* cells (samples) by *p* genes (features)

Options (corresponding script variable):
1. Degrees of freedom (df) - default: 4
2. Kernel Function
    + Matern 1/2, 3/2, 5/2 (m12, m32, m52) - default: True
    + Periodic (per_bool) - default: False
3. Number of Inducing Points (m) - default: 30
4. Batch size (M) - default: 100 *in tGPLVM-iterations-minibatch.py*

Output: hdf5 file with
1. Latent mapping posterior (mean and variance)
2. Gene-specific noise
3. Kernel hyperparameters
