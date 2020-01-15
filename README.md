## Prerequisite
install mujoco-py https://github.com/openai/mujoco-py
install miniconda
add following line to .zshrc
``` export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin:/usr/bin/nvidia-384 ```
```
conda env create -f meta.yaml
conda init zsh
source ~/.zshrc
conda activate NutLearning
conda install scikit-learn
conda install theano
conda install -c conda-forge tensorflow
pip install keras
pip3 install h5py
```

## execution
``` python main.py ```

