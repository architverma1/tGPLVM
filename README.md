# tGPLVM: A robust nonlinear manifold model for single cell RNA-seq data.
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
6. sklearn 0.19.2

### Running
**Input**: A numpy array or sparse csr/csc matrix of scRNA counts (or other types data) with format *N* cells (samples) as rows by *p* genes (features) as columns (loaded to ```y_train```). Input this directly into the code.

**Options**:
The following parameters can be adjusted in the script to adjust inference:

1. Degrees of freedom (```--df```) - default: 4
2. Use t-Distribution error model (otherwise normal error) (```--T```) - default: True
3. Initial Number of Dimensions (```--Q```) - default: 3
4. Kernel Function
    + Matern 1/2, 3/2, 5/2 (```--m12, --m32, --m52```) - default: True
    + Periodic (```--per_bool```) - default: False
5. Number of Inducing Points (```--m```) - default: 30
6. Batch size (```--M```) - default: 250
7. Max iterations (```--iterations```) - default: 5000
8. Save frequency (```--save_freq```): - default: 250
9. Sparse data type (is CSC or CSR) (```--sparse```): - default: False
10. PCA Initialization (otherwise random initialization) (```--pca_init```): - default: True
11. Output directory (```--out```): - default: ./test

**Output**: hdf5 file with
1. Latent mapping posterior (mean and variance)
2. Gene-specific noise
3. Kernel hyperparameters (variance, lengthscale)
4. Inducing points in latent and high-dimensional space

**Example**:

When the input is Test_3_Pollen.h5, the following code runs 250 iterations with the full dataset

```python tGPLVM-batch.py --Q 2 --N 249 --p 6982 --m12 True --m32 True --m52 True --iterations 250 --out ./test```

We provide the input code for two other files:
1. tapio_tcell_tpm.txt - Data from Lonnberg gpfates. Data is available at https://github.com/Teichlab/GPfates
2. 1M_neurons_filtered_gene_bc_matrices_h5.h5 - 1 million 10x mice brains cell. Data is available at https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons. Make sure to set    ```--sparse True``` for this data.

The final data from the paper is available here: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/cd34

