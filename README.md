

# Prototype Inverse Projection (PIP)

This repository is the official implementation of the PIP (In preparation). 

![Untitled.pdf](https://github.com/hookhy/PIP/files/14137157/Untitled.pdf)


## Requirements

To install requirements:

```setup
# Install python environment
conda env create -f environment_gpu.yml 

# Activate environment
conda activate benchmark_gnn
```

## Training & evaluation

To train & evaluate the model(s) in the paper with specified dataset, and model, run this command:

```train
python main_CHDclassification.py --config config_CHDclassification_tmp.json --gpu_id 0 --model ModelName
```




