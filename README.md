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
1. training network
``` python main.py ```

2. load in cpp & predict
- build
```
  cd cpp
  mkdir build
  cd build
  cmake ..
  make -j5
```

- execution
  ```
  cd ..
  g++ -I eigen3 src/main.cpp src/weight.cpp src/utils.cpp -std=c++11 -O3 -Wall -march=native -DNDEBUG -Wextra -Ofast
  ```
