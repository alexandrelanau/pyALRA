# pyALRA

pyALRA is a Python implementation of the Adaptively-thresholded Low Rank Approximation (ALRA) method for single-cell RNA-sequencing data imputation. Original repository for R ALRA implementation: https://github.com/KlugerLab/ALRA and original publication : https://www.nature.com/articles/s41467-021-27729-z.

Our current version of the manuscript is available here : https://www.biorxiv.org/content/10.1101/2025.03.20.644345v1.

## Installation

To install the package, use:
```
git clone https://github.com/alexandrelanau/pyALRA.git
```
Then, install it in your preferred environment:

```
pip install .
```

## Usage

A tutorial is available in /tutorials, with an associated h5ad for testing. 

```python
import anndata
from pyALRA import alra, normalize_data, choose_k

# Load your data into an AnnData object
adata = anndata.read_h5ad('your_data.h5ad')
adata = normalize_data(adata)

# Determine the optimal k
k = choose_k(adata.X)

# Apply ALRA
adata.X = alra(adata.X, k['k'])['A_norm_rank_k_cor_sc']
```
## Benchmarks

Benchmarks against r-ALRA are available on : https://www.biorxiv.org/content/10.1101/2025.03.20.644345v1.

## License

This project is licensed under the MIT License.

## Version

pyALRA was originally developed for python 3.10.16 : scipy=1.14.0, scikit-learn=1.5.2, numpy=1.26.4.

## Archive DOI

You can find the archive DOI from ZENODO here : https://doi.org/10.5281/zenodo.15730914.
