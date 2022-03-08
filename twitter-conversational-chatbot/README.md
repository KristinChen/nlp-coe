# Usage

## Install virtual environment
```console
conda create -n twitter_renv python=3.8 #Please keep python version equals to 3.8
conda activate twitter_renv
conda install --file requirements.txt

# Add virtual environment in jupyter notebook
conda install -c conda-forge ipykernel -y #ipykernel should be installed already
python -m ipykernel install --name=twitter_renv
jupyter notebook #launch jupyter notebook
```

## Update virtual environment
```console
conda list --explicit > requirements.txt
```

Toubleshoot to change encoding to 'ANSI' to read `requirements.txt`. [Reference](https://github.com/conda/conda/issues/9519)

## Prerequisite
[Install Anaconda from here](https://www.anaconda.com/products/individual)

