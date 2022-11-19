# CR-LSO
Code implementation for paper "CR-LSO: Convex Neural Architecture Optimization in the Latent Space of Graph Variational Autoencoder with Input Convex Neural Networks".

## Paper abstract
In neural architecture search (NAS) methods based on latent space optimization (LSO), a deep generative model is trained to embed discrete neural architectures into a continuous latent space. In this case, different optimization algorithms that operate in the continuous space can be implemented to search neural architectures. However, the optimization of latent variables is challenging for gradient-based LSO since the mapping from the latent space to the architecture performance is generally non-convex. To tackle this problem, this paper develops a convexity regularized latent space optimization (CR-LSO) method, which aims to regularize the learning process of latent space in order to obtain a convex architecture performance mapping. Specifically, CR-LSO trains a graph variational autoencoder (G-VAE) to learn the continuous representations of discrete architectures. Simultaneously, the learning process of latent space is regularized by the guaranteed convexity of input convex neural networks (ICNNs). In this way, the G-VAE is forced to learn a convex mapping from the architecture representation to the architecture performance. Hereafter, the CR-LSO approximates the performance mapping using the ICNN and leverages the estimated gradient to optimize neural architecture representations. Experimental results on NAS-Bench-101, NAS-Bench-201 and NAS-Bench-301 show that CR-LSO achieves competitive evaluation results in terms of both computational complexity and architecture performance.

## Code implementation
We use NAS-Bench-101, NAS-Bench-201 and NAS-Bench-301 to veryfy the effectiveness of the proposed CR-LSO. The corresponding codes can be found in folders 'nas_bench_101_experiments', 'nas_bench_201_experiments' and 'nas_bench_301_experiments', respectively. The main packgares we use are pytorch and pytorch-geometric.

Code implementations in all benchmarks are independent, thus they can be run conveniently if one is interested in one of the three benchmarks only. 

### Run CR-LSO in NAS-Bench-101 benchmark
1. Downlowd the format 

'models.py' contains the buding compoments we use in CR-LSO implementation, which includes the ICNN, the semi-supervised GNN predictor, the G-VAE, etc.
''

Before searching an architecture, one should train a graph variational auto-encoder (G-VAE) in a semi-supervised manner by code 'train_gvae_semi_supervised.py'
Then, one can optimize neural architectures in the latent space of G-VAE by code 'crlso.py' with a pretrained G-VAE being saved in 'gvae/gave.pth'.
