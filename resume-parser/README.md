# Usage

## Install virtual environment
```console
conda env create --file environment.yml #a virtual environment named `resume_venv` and it's dependencies will be created

## Rename conda environment
```console
conda create --name i-don't-like-the-old-name --clone resume_venv
conda remove --name resume_venv --all
```

## Add virtual environment in jupyter notebook
python -m ipykernel install --name=resume_venv
jupyter notebook #launch jupyter notebook
```

## Update virtual environment
```console
conda env export > environment.yml 
```

## Prerequisite
[Install Anaconda from here](https://www.anaconda.com/products/individual)

