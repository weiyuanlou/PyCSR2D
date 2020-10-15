



# Cori GPU nodes

Installation notes for Cori GPU nodes

```bash
module load python cuda

conda create -n csr2d -c conda-forge python=3.8 cupy mpmath matplotlib ipykernel

source activate csr2d

python -m ipykernel install --user --name csr2d --display-name CSR2D
```


