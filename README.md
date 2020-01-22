## Prerequisite
1. Install miniconda

https://docs.conda.io/en/latest/miniconda.html

2. Create conda environment
```
  conda env create -f meta.yaml
  conda init zsh
  source ~/.zshrc
  conda activate NutLearning
```
3. Add few more dependencies
```
  conda install scikit-learn
  conda install theano
  conda install tensorflow
  pip3 install keras
  pip3 install h5py
```

## execution
``` python main.py ```

