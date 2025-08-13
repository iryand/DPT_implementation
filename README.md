# ABXI-PyTorch

This is the ***official*** Pytorch implementation of paper "DPT: Dynamic Preference Transfer for Cross-Domain Sequential Recommendation" accepted by CIKM 2025 conference (CIKM'25).

We would like to express our gratitude to [ABXI](https://github.com/DiMarzioBian/ABXI) for providing the foundational code framework. If you are interested in gaining deeper insights into data processing, we highly recommend exploring the source code of ABXI.

## 1. Data
In argument '--data', 'afk' refers to Amazon Food-Kitchen dataset, 'amb' refers to Amazon Movie-Book dataset, and 'abe' refers to Amazon Beauty-Electronics dataset.

Processed data are stored in /data/. If you wanna process your own data, please put the data under /data/raw/, and check the preprocess scripts /utils/preprocess.py.

## 2. Usage
Please check demo.sh on running on different datasets.

### 2.1. Ablation and Hyperparameter

Experiments on ablation and hyperparameters can be conducted by adjusting the value of `T` in the `demo.sh` file. When `T=0`, it represents the "Cross" variant mentioned in the paper. When `T=1e-9`, it represents the "Self" variant. Other values of `T` correspond to the temperature settings used in the experiments described in the paper. You can set different values of `T` based on the requirements of different datasets.


## 3. Citation

If you find our code helpful, please cite our paper. The link to the paper will be provided as soon as it is published.


## 4. File Tree

    DPT/
    ├── data/
    │   ├── abe/
    │   │   ├── abe_50_preprocessed.txt
    │   │   ├── abe_50_seq.pkl
    │   │   ├── map_item.txt
    │   │   └── map_user.txt
    │   ├── afk/
    │   │   ├── afk_50_preprocessed.txt
    │   │   ├── afk_50_seq.pkl
    │   │   ├── map_item.txt
    │   │   └── map_user.txt
    │   └── amb/
    │       ├── amb_50_preprocessed.txt
    │       ├── map_item.txt
    │       └── map_user.txt
    ├── dataloader.py
    ├── demo.sh
    ├── main.py
    ├── models/
    │   ├── DPT.py
    │   └── encoders.py
    ├── README.md
    ├── requirements.txt
    ├── trainer.py
    └── utils/
        ├── constants.py
        ├── metrics.py
        ├── misc.py
        ├── noter.py
        └── preprocess.py


Because the files 'amb_50_seq.pkl' exceeds Github file size limit, you have to manually generate this preprocessed serialized data first, by adding argument '--raw' in the first experiment on Movie-Book dataset.