# tGPLVM: A Nonparametric, Generative Model for Manifold Learning with scRNA-seq experimental data
## Intro

Dimension reduction is a common and critical first step in analysis of high throughput singe cell RNA sequencing. tGPLVM is the a nonparametric, generative model for nonlinear manifold learning; that is a flexible, nearly assumption-free model that doesn't require setting parameters *a priori* (e.g. number of dimensions, perplexity, etc.) and provides uncertainty estimates for sample mappings. tGPLVM can be used for visualization of high-dimensional data or as part of a pipeline for cell type identification or pseudotime reconstruction. 

We provide a script for fitting the model with Black Box Variational Inference for speed and scabality. A batch learning implementation is also provided for larger datasets that need to be fit under memory restriction.

## Usage
tGPLVM is implemented in python 2.7 with the following packages:
1. numpy
2. pandas
3. h5py
4. tensorflow
5. edwards
6. sklearn

Input: A csv or hdf5 of scRNA counts (or other types data) with format *N* cells (samples) as rows by *p* genes (features) as columns

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
