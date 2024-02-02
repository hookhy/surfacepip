

# Prototype Inverse Projection (PIP)

This repository is the official implementation of the PIP (In preparation). 

![Untitled](https://github.com/hookhy/PIP/assets/84267304/8fe4f5d8-1f42-4bee-bc20-6f7cb4274c0a)


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




