# Usage

## Method 1: Install virtual environment using `environment.yml` (recommended)
### 1. Install virtual environment
```console
conda env create --file environment.yml #a virtual environment named `chatbot_venv` and it's dependencies will be created
```

### 2. Rename conda environment
```console
conda create --name i-don't-like-the-old-name --clone chatbot_venv
conda remove --name chatbot_venv --all
```

### 3. Add virtual environment in jupyter notebook
```console
python -m ipykernel install --name=chatnot_venv
jupyter notebook #launch jupyter notebook after adding it into your Path
```
[How to launch your jupyter notebook from terminal](https://towardsdatascience.com/how-to-launch-jupyter-notebook-quickly-26e500ad4560)

### 4. Update virtual environment
```console
conda env export > environment.yml 
```

## Method 2: Install virtual environment using `requirements.txt`
### 1. Install virtual environment
```console
conda create -n twitter_renv python=3.8 #Please keep python version equals to 3.8
conda activate twitter_renv
conda install --file requirements.txt
```

### 2. Add virtual environment in jupyter notebook
python -m ipykernel install --name=twitter_renv
jupyter notebook #launch jupyter notebook
```

### 3. Update virtual environment
```console
conda list --explicit > requirements.txt
```

Toubleshoot to change encoding to 'ANSI' to read `requirements.txt`. [Reference](https://github.com/conda/conda/issues/9519)

## Prerequisite
[Install Anaconda from here](https://www.anaconda.com/products/individual)

