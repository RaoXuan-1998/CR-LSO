# CR-LSO
Code implementation for paper "CR-LSO: Convex Neural Architecture Optimization in the Latent Space of Graph Variational Autoencoder with Input Convex Neural Networks".
![image](https://github.com/RaoXuan-1998/CR-LSO/blob/main/pictures/CR-LSO.jpg)

## Paper abstract
In neural architecture search (NAS) methods based on latent space optimization (LSO), a deep generative model is trained to embed discrete neural architectures into a continuous latent space. In this case, different optimization algorithms that operate in the continuous space can be implemented to search neural architectures. However, the optimization of latent variables is challenging for gradient-based LSO since the mapping from the latent space to the architecture performance is generally non-convex. To tackle this problem, this paper develops a convexity regularized latent space optimization (CR-LSO) method, which aims to regularize the learning process of latent space in order to obtain a convex architecture performance mapping. Specifically, CR-LSO trains a graph variational autoencoder (G-VAE) to learn the continuous representations of discrete architectures. Simultaneously, the learning process of latent space is regularized by the guaranteed convexity of input convex neural networks (ICNNs). In this way, the G-VAE is forced to learn a convex mapping from the architecture representation to the architecture performance. Hereafter, the CR-LSO approximates the performance mapping using the ICNN and leverages the estimated gradient to optimize neural architecture representations. Experimental results on NAS-Bench-101, NAS-Bench-201 and NAS-Bench-301 show that CR-LSO achieves competitive evaluation results in terms of both computational complexity and architecture performance.

## Code implementation
1. We use NAS-Bench-101, NAS-Bench-201 and NAS-Bench-301 to veryfy the effectiveness of the proposed CR-LSO. The corresponding codes can be found in folders 'nas_bench_101_experiments', 'nas_bench_201_experiments' and 'nas_bench_301_experiments', respectively. The main packgares we use are pytorch and pytorch-geometric.

2. Code implementations in all benchmarks are independent, thus they can be run conveniently if one is interested in one of the three benchmarks only. 

### Run CR-LSO in NAS-Bench-101 benchmark
1. Download the json-format NAS-Bench-101 dataset used in paper [Generalized Global Ranking-Aware Neural Architecture Ranker for Efficient
Image Classifier Search](https://arxiv.org/pdf/2201.12725.pdf) from [here](https://github.com/AlbertiPot/nar). We do not use the orginal [NAS-Bench-101](https://github.com/google-research/nasbench) benchmark is too heavy to download. After downloading the json-format dataset, put it in 'data/nasbench_only108_with_vertex_flops_and_params.json'.
2. Convert the json-format dataset to a geometric-like dataset using code 'collect_101_dataset.py'. The transformation is needed since we use torch-geometric to build the GNN predictor and G-VAE. After transformation, the dataset will be saved in 'dataset/nas_bench_101.pth'.
3. Train a G-VAE in a semi-supervised manner using code 'train_gvae_semi_supervised.py'. The trained G-VAE will be saved in 'gave/gvae.pth'.
4. Optimize neural architectures in the latent space of G-VAE by code 'crlso.py' with a pretrained G-VAE.

### Run CR-LSO in NAS-Bench-201 benchmark
1. Download the json-format NAS-Bench-201 dataset used in paper [Generalized Global Ranking-Aware Neural Architecture Ranker for Efficient
Image Classifier Search](https://arxiv.org/pdf/2201.12725.pdf) from [here](https://github.com/AlbertiPot/nar). After downloading the json-format dataset, put it in 'data/nasbench201_with_edge_flops_and_params.json'.
2. The remaining steps are the same as those in NAS-Bench-101 experiments.

### Run CR-LSO in NAS-Bench-301 benchmark
1. Coming soon... (will be released after the paper is accpeted)
