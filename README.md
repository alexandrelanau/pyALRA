# pyALRA

pyALRA is a Python implementation of the Adaptively-thresholded Low Rank Approximation (ALRA) method for single-cell RNA-sequencing data imputation.

## Installation

To install the package, use:

```
pip install .
```

## Usage

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

## License

This project is licensed under the MIT License.
