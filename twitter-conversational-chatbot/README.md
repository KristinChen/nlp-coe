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

### 5. Remove virtual environment
```console
conda env remove --name chatbot_renv
```

