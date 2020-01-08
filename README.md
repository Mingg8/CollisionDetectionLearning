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
```

## execution
``` python fcl_learning_by_demonstration.py ```

