# tGPLVM: A Nonparametric, Generative Model for Manifold Learning with scRNA-seq experimental data
## Intro

Dimension reduction is a common and critical first step in analysis of high throughput singe cell RNA sequencing. tGPLVM is a nonparametric, generative model for nonlinear manifold learning; that is a flexible, nearly assumption-free model that doesn't require setting parameters *a priori* (e.g. number of dimensions, perplexity, etc.) and provides uncertainty estimates for sample mappings. tGPLVM can be used for visualization of high-dimensional data or as part of a pipeline for cell type identification or pseudotime reconstruction. 

We provide a script for fitting the model with Black Box Variational Inference for speed and scabality. A batch learning implementation is also provided for larger datasets that need to be fit under memory restriction.

## Usage

### Requirements

tGPLVM is implemented in python 2.7 with the following packages:
1. numpy 1.14.5
2. pandas 0.23.3
3. h5py 2.8.0
4. tensorflow 1.6.0
5. edwards 1.3.5
6. sklearn 0.0

### Running
**Input**: A numpy array of scRNA counts (or other types data) with format *N* cells (samples) as rows by *p* genes (features) as columns (loaded to ```y_train```)

**Options** (corresponding script variable):
The following parameters can be adjusted in the script to adjust inference:

1. Degrees of freedom (```df```) - default: 4
2. Initial Number of Dimensions (```Q```) - default: 3
3. Kernel Function
    + Matern 1/2, 3/2, 5/2 (```m12, m32, m52```) - default: True
    + Periodic (```per_bool```) - default: False
4. Number of Inducing Points (```m```) - default: 30
5. Batch size (```M```) - default: 100 (*in tGPLVM-iterations-minibatch.py*)
6. Max iterations (```iterations```) - default: 5000
7. Save frequency (```save_freq```): - default: 250

**Output**: hdf5 file with
1. Latent mapping posterior (mean and variance)
2. Gene-specific noise
3. Kernel hyperparameters (variance, lengthscale)
4. Inducing points in latent and high-dimensional space
5. Latent high-dimensional data (denoised data)


