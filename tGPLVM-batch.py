import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd

from edward.models import MultivariateNormalTriL, Normal, StudentT
from tensorflow.contrib.distributions import softplus_inverse

from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

from scipy.sparse import csc_matrix
from scipy.io import mmread

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import shutil, os, subprocess
import h5py
import time

import argparse
# settings

parser = argparse.ArgumentParser(description = 'This script maps scRNA seq data to a low dimensional manifold using a modified GPLVM')

parser.add_argument('--out', dest = 'outdir', type = str, default = './test', help = 'output directory')

parser.add_argument('--Q', dest = 'Q', type = int, default = 3, help = 'Initial number of latent dimensions')

parser.add_argument('--m12', dest = 'm12', type = bool, default = False, help = 'Include Matern 1/2 kernel')
parser.add_argument('--m32', dest = 'm32', type = bool, default = False, help = 'Include Matern 3/2 kernel')
parser.add_argument('--m52', dest = 'm52', type = bool, default = False, help = 'Include Matern 5/2 kernel')
parser.add_argument('--T', dest = 'Terror', type = bool, default = True, help = 'Use Student t Error')

#parser.add_argument('--in_path', dest = 'in_path', type = str, help = 'mtx file')

args = parser.parse_args()

ns = 1000000
SIMULATED = False
PCA_INIT = True
CELLS_BY_GENES = True
SPARSE = True

m12 = args.m12
m32 = args.m32
m52 = args.m52
per_bool = False

Terror = args.Terror

Q = args.Q
m = 30
df = 4.0
offset = 0.01
M = np.minimum(ns,2500)
p = 250


iterations = 100 * int(ns/M)
print(iterations)
save_freq = 500
#out_dir = './cord-blood/t-error/q5/'
#out_dir = './test_kern_fx/' # save to scratch
out_dir = args.outdir
# create out directory
if os.path.exists(out_dir):
	print "Output directory %s already exists.  Removing it to have a clean output directory!" % out_dir
	shutil.rmtree(out_dir)
os.makedirs(out_dir)

# load data


#dat_path = '/home/architv/single-cell/tGPLVM/tGPLVM/dat/trajectory/y_train_trajectory.csv'
#y_train = np.loadtxt(dat_path)

#dat_path = '/home/architv/single-cell/tGPLVM/tGPLVM/dat/trajectory/y_train_trajectory.csv'
#y_train = np.loadtxt(dat_path)

#xp = pd.read_csv('/home/architv/single-cell/tGPLVM/tGPLVM/dat/splatter-groups.csv')
#y_train = xp.values[:,1:].T.astype(np.float32)

#dat_waterfall = pd.read_csv('/home/archithpc/data/GSE71485_Single_TPM.txt', delimiter = '\t')
#y_train = dat_waterfall.values.T #[np.sum(dat_waterfall.values,axis = 1) > 10].T
#y_train = np.log2(1+y_train)

#dat_path = '/home/archithpc/sc-dim-red/Test_3_Pollen.h5'
#dat_file = h5py.File(dat_path,'r')
#y_train = dat_file['in_X'][:]

dat_path = '1M_neurons_filtered_gene_bc_matrices_h5.h5'
dat_file = h5py.File(dat_path, 'r')
y_train = csc_matrix((dat_file['mm10']['data'],dat_file['mm10']['indices'],dat_file['mm10']['indptr']))
y_train = y_train.T[:ns,:]

#dat_path = '/home/archithpc/data/ica_bone_marrow_h5.h5'
#dat_file = h5py.File(dat_path, 'r')
#y_train = csc_matrix((dat_file['GRCh38']['data'],dat_file['GRCh38']['indices'],dat_file['GRCh38']['indptr']))
#y_train = y_train.T

#dat_path = '/home/archithpc/data/t_3k_4k_aggregate_filtered_gene_bc_matrices_h5.h5'
#dat_file = h5py.File(dat_path, 'r')
#y_train = csc_matrix((dat_file['GRCh38']['data'],dat_file['GRCh38']['indices'],dat_file['GRCh38']['indptr']))
#y_train = y_train.T

#d2 = pd.read_csv('/home/archithpc/data/tapio_tcell_tpm.txt', delimiter = '\t')
#y_train = d2.values[(np.sum(d2.values[:,1:],axis = 1) > 0),1:].astype(np.float32)
#y_train = np.log2(1+y_train.T)

#dat_path = '/home/archithpc/data/chorion.txt'
#dat = pd.read_csv(dat_path, delimiter = '\t')
#y_train = dat.values[:,1:].T.astype(np.float32)
#y_train = np.log2(1.+y_train)

#dat_path = '/home/archithpc/data/mouse-2k-nuerons/mm10/matrix.mtx'
#dat_path = '/home/archithpc/data/donor-a-68k-pbmcs/filtered_matrices_mex/hg19/matrix.mtx'
#dat_path = args.in_path
#filtered = mmread(dat_path)
#y_train = csc_matrix(filtered).T


if CELLS_BY_GENES:
	N = y_train.shape[0]
	G = y_train.shape[1]
else:
	N = y_train.shape[1]
	G = y_train.shape[0]
	y_train = y_train.T

#gene_weights = 200.*np.sum(y_train > 0, axis = 0)/float(N)
if SPARSE:
	exp = y_train > 0
	exp_weight = exp.sum(axis = 0)
	print(exp_weight.shape)
	gene_weights = 200.* np.squeeze(np.array(exp_weight), axis = 0)/float(N)
else:
	gene_weights = 200.*np.sum(y_train > 0, axis = 0)/float(N)
p_g = np.exp(gene_weights)/np.sum(np.exp(gene_weights))
#p_g = gene_weights/float(np.sum(gene_weights))
print(gene_weights.shape)
print(p_g.shape)

if SIMULATED:
	true_path = '/home/architv/single-cell/tGPLVM/tGPLVM/dat/trajectory/x_true_trajectory.csv'
	x_true = np.loadtxt(true_path)
	Q_true = x_true.shape[1]
else:
	x_true = np.zeros((N,Q))
	Q_true = Q

print('Data Size: ' + str(N) + ' cells by ' + str(G) + ' genes')
print('Bath Size: ' + str(M) + ' cells by ' + str(p) + ' genes')


# Batch drawing fx
def next_batch(x_train, M, p):
    idx_batch = np.random.choice(N, M, replace = False)
    xslice1 = x_train[idx_batch]
    gene_ix = np.random.choice(G, size = p,replace = False, p = p_g)
    #gene_ix = np.array(range(0,p))
    xslice2 = xslice1[:,gene_ix].astype(np.float64)
    if SPARSE:
            xslice2 = xslice2.toarray()
	    xslice2 = np.log2(1. + xslice2)
    return xslice2, idx_batch.astype(np.int32), gene_ix.astype(np.int32)

### Inference

## Kernel functions

def rbf(X1,X2,lengthscale,variance):
	X1l = X1/lengthscale
	X2l = X2/lengthscale
	X1s = tf.reduce_sum(tf.square(X1l),1)
	X2s = tf.reduce_sum(tf.square(X2l),1)
	square = tf.reshape(X1s, [-1,1]) + tf.reshape(X2s, [1,-1]) - 2 * tf.matmul(X1l,X2l,transpose_b = True)
	K = variance * tf.exp(-square/2)
	return K

def matern12(X1,X2,lengthscale,variance):
	X1l = X1/lengthscale
        X2l = X2/lengthscale
	X1s = tf.reduce_sum(tf.square(X1l),1)
	X2s = tf.reduce_sum(tf.square(X2l),1)
	square = tf.reshape(X1s, [-1,1]) + tf.reshape(X2s, [1,-1]) - 2 * tf.matmul(X1l,X2l,transpose_b = True)
	K = variance * tf.exp(-tf.abs(square))
	return K

def matern32(X1,X2,lengthscale,variance):
    X1l = X1/lengthscale
    X2l = X2/lengthscale
    X1s = tf.reduce_sum(tf.square(X1l),1)
    X2s = tf.reduce_sum(tf.square(X2l),1)
    square = tf.reshape(X1s, [-1,1]) + tf.reshape(X2s, [1,-1]) - 2 * tf.matmul(X1l,X2l,transpose_b = True)
    K = variance * (1. + np.sqrt(3.)*tf.abs(square)) * tf.exp(-np.sqrt(3.) * tf.abs(square))
    return K

def matern52(X1,X2,lengthscale,variance):
    X1l = X1/lengthscale
    X2l = X2/lengthscale
    X1s = tf.reduce_sum(tf.square(X1l),1)
    X2s = tf.reduce_sum(tf.square(X2l),1)
    square = tf.reshape(X1s, [-1,1]) + tf.reshape(X2s, [1,-1]) - 2 * tf.matmul(X1l,X2l,transpose_b = True)
    K = variance * (1. + np.sqrt(5.)*tf.abs(square) + 5./3. * tf.square(square)) * tf.exp(-np.sqrt(5.) * tf.abs(square))
    return K

def periodic(X1,X2,variance,period):
    X1l = X1/period
    X2l = X2/period
    X1s = tf.reduce_sum(tf.square(X1l),1)
    X2s = tf.reduce_sum(tf.square(X2l),1)
    square = tf.reshape(X1s, [-1,1]) + tf.reshape(X2s, [1,-1]) - 2 * tf.matmul(X1l,X2l,transpose_b = True)
    #sin_square = tf.square(tf.sin(2*np.pi*square)/lengthscale)
    K = variance * tf.exp(-0.5*tf.square(tf.sin(np.pi*square)))
    return K

def kernelfx(X1,X2): # takes float32, casts to float64, computes float64 kernel
    X1_64 = tf.cast(X1, dtype = tf.float64)
    X2_64 = tf.cast(X2, dtype = tf.float64)
    K = rbf(X1_64,X2_64,tf.cast(lengthscale,dtype=tf.float64),tf.cast(variance,tf.float64))
    if m12:
        K += matern12(X1_64,X2_64,tf.cast(lengthscaleM12,dtype = tf.float64),tf.cast(varianceM12,tf.float64))
    if m32: 
        K += matern32(X1_64,X2_64,tf.cast(lengthscaleM32,dtype = tf.float64),tf.cast(varianceM32,tf.float64))
    if m52:
        K += matern52(X1_64,X2_64,tf.cast(lengthscaleM52,dtype = tf.float64),tf.cast(varianceM52,tf.float64))
    if per_bool:
        K += periodic(X1_64,X2_64,tf.cast(period_var,dtype = tf.float64),tf.cast(period,dtype = tf.float64))
    return K

## Initialization

# Initialize latent space X with random noise or PCA

if PCA_INIT:
	if SPARSE:
		pca = NMF(n_components = Q)
		#pca = PCA(n_components = Q)
	else:
		pca = PCA(n_components = Q)
	qx_init = pca.fit_transform(y_train) #.tocsr())
	qx_init = 15.*qx_init/np.max(qx_init) 
	#print(np.max(qx_init))
else:
	qx_init = np.random.normal(0,1,size = [N,Q])

# Initialize lengths with max distance across dimensions
len_init = np.abs(np.max(qx_init, axis = 0) - np.mean(qx_init,axis = 0))

# Initialize X_u with random subset of initalization
xu_init = np.random.choice(N, size = m)
xu = tf.Variable(qx_init[xu_init], dtype = tf.float64)
qu_init = y_train[xu_init] 

## Define p(x)

lengthscale_pre = tf.Variable(np.log(np.exp(len_init)-1), dtype = tf.float32)
variance_pre = tf.Variable(np.log(np.exp(0.5)-1), dtype = tf.float32)

lengthscale = tf.nn.softplus(lengthscale_pre)
variance = tf.nn.softplus(variance_pre)

lengthscaleM12_pre = tf.Variable(np.log(np.exp(len_init)-1), dtype = tf.float32)
varianceM12_pre = tf.Variable(np.log(np.exp(0.5)-1), dtype = tf.float32)

lengthscaleM12 = tf.nn.softplus(lengthscaleM12_pre)
varianceM12 = tf.nn.softplus(varianceM12_pre)

lengthscaleM32_pre = tf.Variable(np.log(np.exp(7.*len_init)-1), dtype = tf.float32)
varianceM32_pre = tf.Variable(np.log(np.exp(0.1)-1), dtype = tf.float32)

lengthscaleM32 = tf.nn.softplus(lengthscaleM32_pre)
varianceM32 = tf.nn.softplus(varianceM32_pre)

lengthscaleM52_pre = tf.Variable(np.log(np.exp(7.*len_init)-1), dtype = tf.float32)
varianceM52_pre = tf.Variable(np.log(np.exp(0.1)-1), dtype = tf.float32)

lengthscaleM52 = tf.nn.softplus(lengthscaleM52_pre)
varianceM52 = tf.nn.softplus(varianceM52_pre)


#period_pre = tf.Variable(np.log(np.exp(7.0*len_init)-1), dtype = tf.float32)
#period_len_pre = tf.Variable(1.0)
#period_var_pre = tf.Variable(np.log(np.exp(0.5)-1), dtype = tf.float32)#

#period = tf.nn.softplus(period_pre)
#period_length = tf.nn.softplus(period_len_pre)
#period_var = tf.nn.softplus(period_var_pre)

#Kuu = rbf(xu,xu,tf.cast(lengthscale,dtype=tf.float64),tf.cast(variance,tf.float64)) + \
#matern12(xu,xu,tf.cast(lengthscaleM12,dtype = tf.float64),tf.cast(varianceM12,tf.float64)) # + \
#matern32(xu,xu,tf.cast(lengthscaleM32,dtype = tf.float64),tf.cast(varianceM32,tf.float64)) + \
#matern52(xu,xu,tf.cast(lengthscaleM52,dtype = tf.float64),tf.cast(varianceM52,tf.float64)) # + \
#periodic(xu,xu,tf.cast(period_var,dtype = tf.float64),tf.cast(period,dtype = tf.float64))'''

Kuu = kernelfx(xu,xu)

fu_loc = tf.zeros((p,m))
fu_scale = tf.cast(tf.cholesky(Kuu + offset*tf.eye(m, dtype = tf.float64), name = 'fu_scale'), dtype = tf.float32)

u = MultivariateNormalTriL(loc = fu_loc, scale_tril = fu_scale, name = 'pu')
x = Normal(loc = tf.zeros((M,Q)), scale = 1.0)


#xu32 = tf.cast(xu,dtype = tf.float32)

#Kfu = rbf(x,xu32,lengthscale,variance) + matern12(x,xu32,lengthscaleM12,varianceM12)# + matern32(x,xu32,lengthscaleM32,varianceM32) + matern52(x,xu32,lengthscaleM52,varianceM52) #+ periodic(x,xu32,period_var,period)

Kfu = kernelfx(x,xu)

#Kff = rbf(x,x,lengthscale,variance) + matern12(x,x,lengthscaleM12,varianceM12)# + matern32(x,x,lengthscaleM32,varianceM32) + matern52(x,x,lengthscaleM52,varianceM52)# + periodic(x,x,period_var,period)

Kff = kernelfx(x,x)

Kuuinv = tf.matrix_inverse(Kuu + offset*tf.eye(m, dtype = tf.float64))
KfuKuuinv = tf.matmul(Kfu, Kuuinv)
KffKuuinvU = [tf.reshape(tf.matmul(KfuKuuinv, tf.expand_dims(tf.cast(u[i], dtype = tf.float64), axis = 1)), [-1]) for i in range(0,p)]

KffKuuKuf = tf.matmul(KfuKuuinv, Kfu, transpose_b = True)
sigmaf_temp = Kff - KffKuuKuf
sigmaf_diag = tf.matrix_band_part(sigmaf_temp, 0, 0)
sigmaf_upperT = tf.matrix_band_part(sigmaf_temp, 0, -1)
sigmaf = sigmaf_upperT + tf.transpose(sigmaf_upperT) - sigmaf_diag
f_scale = tf.cholesky(sigmaf + offset*tf.eye(M, dtype = tf.float64), name = 'f_scale')


# p(F|U,X,Xu)
f = MultivariateNormalTriL(loc = tf.cast(KffKuuinvU, dtype = tf.float32), scale_tril = tf.cast(f_scale, dtype = tf.float32), name = 'pf')

# p(Y|F)
t_var_pre = tf.Variable(0.5*np.ones((G,1)), dtype = tf.float32)
t_var_full = tf.nn.softplus(t_var_pre)
idx_g = tf.placeholder(tf.int32, p)
t_var = tf.gather(t_var_full, idx_g)
if Terror:
	y = StudentT(df = df, loc = f, scale = t_var)
else:
	y = Normal(loc = f, scale = t_var)

## Define q(x)

qx_mean = tf.Variable(qx_init, dtype = tf.float32, name = 'qx_mean')
qx_scale = tf.Variable(tf.ones((N,Q)), dtype = tf.float32, name = 'qx_scale')

idx_ph = tf.placeholder(tf.int32, M)
qx_mini = tf.gather(qx_mean, idx_ph)
qx_scale_mini = tf.gather(qx_scale, idx_ph)

qx = Normal(loc = qx_mini, scale = tf.nn.softplus(qx_scale_mini), name = 'qx')

#QKfu = rbf(qx,xu32,lengthscale, variance) + matern12(qx,xu32,lengthscaleM12, varianceM12)# + matern32(qx,xu32,lengthscaleM32, varianceM32) + matern52(qx,xu32,lengthscaleM52, varianceM52)# + periodic(qx,xu32,period_var,period)
QKfu = kernelfx(qx,xu)

#QKff = rbf(qx,qx,lengthscale,variance) + matern12(qx,qx,lengthscaleM12,varianceM12)# + matern32(qx,qx,lengthscaleM32,varianceM32) + matern52(qx,qx,lengthscaleM52,varianceM52)# + periodic(qx,qx,period_var,period)

QKff = kernelfx(qx,qx)

KuuPsi = tf.matrix_inverse(tf.matmul(QKfu,QKfu,transpose_a = True) + offset*tf.eye(m, dtype = tf.float64), name = 'Kuu_Psi')
KuuKuuPsi = tf.matmul(Kuu, KuuPsi)
y_dat = tf.placeholder(tf.float64, [M,p])
y_dat_ph = tf.cast(tf.transpose(y_dat), dtype = tf.float32)
PsiY = tf.matmul(QKfu,y_dat, transpose_a = True)
qu_loc = tf.transpose(tf.matmul(KuuKuuPsi,PsiY))
qu = MultivariateNormalTriL(loc = tf.cast(qu_loc, dtype = tf.float32), scale_tril = tf.cast(fu_scale, dtype = tf.float32), name = 'qu')

QKfuKuuinv = tf.matmul(QKfu, Kuuinv)
QKfuKuuinvU = [tf.reshape(tf.matmul(QKfuKuuinv, tf.expand_dims(tf.cast(qu[i], dtype = tf.float64), axis = 1)), [-1]) for i in range(0,p)]

QKfuKuuKuf = tf.matmul(QKfuKuuinv, QKfu, transpose_b = True)
Qsigmaf_temp = QKff - QKfuKuuKuf
Qsigmaf_diag = tf.matrix_band_part(Qsigmaf_temp, 0, 0)
Qsigmaf_upperT = tf.matrix_band_part(Qsigmaf_temp, 0, -1)
Qsigmaf = Qsigmaf_upperT + tf.transpose(Qsigmaf_upperT) - Qsigmaf_diag
Qf_scale = tf.cholesky(Qsigmaf + offset*tf.eye(M, dtype = tf.float64), name = 'qf_scale')

qf = MultivariateNormalTriL(loc = tf.cast(QKfuKuuinvU, dtype = tf.float32), scale_tril = tf.cast(Qf_scale, dtype = tf.float32), name = 'qf')


## Save fx
def save(name):
    time_now = time.time()
    elapsed = time_now - time_start
    x_post = qx_mean.eval()
    x_var_post = tf.nn.softplus(qx_scale).eval()
    #f_post = qf.eval(feed_dict = {y_dat: y_train.T, idx_ph: range(0,N)})
    xu_post = xu.eval()
    u_post = qu.loc.eval(feed_dict = {y_dat: y_batch, idx_ph: idx_batch, idx_g: gix})
    t_scales = t_var_full.eval()

    len_rbf = np.expand_dims(lengthscale.eval(),axis = 1)
    var_rbf = variance.eval()
    len_12 = np.expand_dims(lengthscaleM12.eval(), axis = 1)
    var_12 = varianceM12.eval()
    len_32 = np.expand_dims(lengthscaleM32.eval(), axis = 1)
    var_32 = varianceM32.eval()
    len_52 = np.expand_dims(lengthscaleM52.eval(), axis = 1)
    var_52 = varianceM52.eval()
    
    kern_names = np.array(['RBF','1/2','3/2','5/2'])
    kern_lengths = np.concatenate((len_rbf,len_12,len_32,len_52),axis = 1)
    kern_vars = np.array([var_rbf,var_12,var_32,var_52])

    fname = os.path.join(out_dir, name)
    fout = open(fname, 'w+')
    fout.close()

    fout = h5py.File(fname,'w')

    #fout.create_dataset("y_train", data = y_train)
    fout.create_dataset("x_mean", data = x_post)
    fout.create_dataset("x_var", data = x_var_post)
    #fout.create_dataset("f", data = f_post)
    fout.create_dataset("xu", data = xu_post)
    fout.create_dataset("u", data = u_post)
    fout.create_dataset("scales", data = t_scales)
    fout.create_dataset("kernel_names", data = kern_names)
    fout.create_dataset("lengthscales", data = kern_lengths)
    fout.create_dataset("variances", data = kern_vars)
    fout.create_dataset("elapsed_time", data = elapsed)

    if SIMULATED:
        fout.create_dataset("x_true", data = x_true)


    fout.close()

## Run Inference
inference = ed.KLqp({f: qf, x: qx, u: qu}, data = {y: y_dat_ph})
#inference.run(n_iter = iterations, logdir = 'log/inducing_pts')
time_start = time.time()
inference.initialize()
tf.global_variables_initializer().run()

test_batch, test_idx, gix = next_batch(y_train,M, p)

info_dict = inference.update(feed_dict = {y_dat: test_batch, idx_ph: test_idx, idx_g: gix})
inference.print_progress(info_dict)
loss_old = info_dict['loss'];

convergence_counter = 0
test_freq = 15
min_iter = 100

chol_fails = 0

for iteration in range(1,iterations):
    #if iteration % test_freq == 0:
#	y_batch = test_batch
#	idx_batch = test_idx
#    else:
#	y_batch, idx_batch = next_batch(y_train, M)
    
    y_batch, idx_batch, gix = next_batch(y_train, M, p)
    try: 
    	info_dict = inference.update(feed_dict = {y_dat: y_batch, idx_ph: idx_batch, idx_g: gix})
	inference.print_progress(info_dict)
    except:
    	print('chol_error')
	offset += .01
        chol_fails +=1
   
    if iteration % save_freq == 0:
        temp_name = 'model-output-' + str(iteration) + '.hdf5'
        save(temp_name)

#    if iteration % test_freq == 0:
#	    loss_new = info_dict['loss']
#	    if loss_new > loss_old:
#		    print(iteration)
#		    convergence_counter += 1
#		    print(convergence_counter)
#	    else:
#		    convergence_counter = 0
#		    loss_old = loss_new
#    if iteration > min_iter:
#	    loss_new = info_dict['loss']
#	    if loss_new > loss_old:
#		    convergence_counter += 1
#	    else:
#		    convergence_counter = 0
#		    loss_old = loss_new
#	    if convergence_counter > 5:
#		    break
inference.finalize()
print(chol_fails)
## Save Results
save('model-output-final.hdf5')









