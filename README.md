# Understanding Normalization in Contrastive Representation Learning and Out-of-Distribution Detection

This repository contains the implementation for OECL.

**Abstract**
Contrastive representation learning has emerged as an outstanding approach for anomaly detection. In this work, we explore the $\ell_2$-norm of contrastive features and its applications in out-of-distribution detection. We propose a simple method based on contrastive learning, which incorporates out-of-distribution data by discriminating against normal samples in the contrastive layer space. Our approach can be applied flexibly as an outlier exposure (OE) approach, where the out-of-distribution data is a huge collective of random images, or as a fully self-supervised learning approach, where the out-of-distribution data is self-generated by applying distribution-shifting transformations. The ability to incorporate additional out-of-distribution samples enables a feasible solution for datasets where AD methods based on contrastive learning generally underperform, such as aerial images or microscopy images. Furthermore, the high-quality features learned through contrastive learning consistently enhance performance in OE scenarios, even when the available out-of-distribution dataset is not diverse enough.
Our extensive experiments demonstrate the superiority of our proposed method under various scenarios, including unimodal and multimodal settings, with various image datasets.

## Training and evaluating
By default, all code examples are assuming distributed launch with 4 multi GPUs. The number of GPUs can be changed at file run*.sh
### Train and evaluate one vs. rest (unimodal) with OECL
```
cd OECL/src
bash run.sh
```
or
```
cd OECL/src
torchrun --standalone --nnodes=1 --nproc_per_node=4 ./main/oecl/oecl.py --config_env ./yaml/oecl/env_oecl.yaml --config_exp ./yaml/oecl/oecl.yaml  --times 1 --seed $RANDOM --ddp True --id_class <id_class>
```
Datasets, batch_size and other settings can be changed at OECL/src/yaml/oecl/oecl.yaml.

### Train and evaluate leave-one-class-out (multimodal) with OECL
```
cd OECL/src
bash run_m.sh
```
or 
```
cd OECL/src
torchrun --standalone --nnodes=1 --nproc_per_node=4 ./main/oecl_m/oecl.py --config_env ./yaml/oecl_m/env_oecl.yaml --config_exp ./yaml/oecl_m/oecl.yaml  --times 1 --seed $RANDOM --ddp True --id_class <id_class>
```

Datasets, batch_size and other settings can be changed in OECL/src/yaml/oecl_m/oecl.yaml.

### Train and evaluate one vs. rest (unimodal) with self-OECL
```
cd OECL/src
bash run_self.sh
```
or 
```
cd OECL/src
torchrun --standalone --nnodes=1 --nproc_per_node=4 ./main/oecl_self/oecl.py --config_env ./yaml/oecl_self/env.yaml --config_exp ./yaml/oecl_self/oecl.yaml  --times 1 --seed $RANDOM --ddp True --id_class <id_class>
```

Datasets, batch_size and other settings can be changed in OECL/src/yaml/oecl_self/oecl.yaml.

**Results**
### One vs. Rest
| Method           | Dataset  | OE dataset | AUROC |
|------------------|----------|------------|-------|
| CSI              | CIFAR-10 | [x]        | 94.3  |
| self-OECL (ours) | CIFAR-10 | [x]        | 94.7  |
| OECL (ours)      | CIFAR-10 | Tiny80m    | 97.8  |
| HSC              | CIFAR-10 | Tiny80m    | 95.9  |

We only show one vs. rest CIFAR-10 results in this repo. For other settings, please see our paper.

### Leave-one-class-out

| Method      | Dataset  | OE Dataset | AUROC |
|-------------|----------|------------|-------|
| SimCLR      | CIFAR-10 | [x]        | 83.9  |
| HSC         | CIFAR-10 | Tiny80m    | 84.8  |
| BCE         | CIFAR-10 | Tiny80m    | 86.6  |
| OECL (ours) | CIFAR-10 | Tiny80m    | 94.6  |

We only show one vs. rest CIFAR-10 results in this repo. For other settings, please see our paper.

*For other benchmarks like few-shot OOD benchmark, please see our paper*

# Citation
```
@misc{legia2023understanding,
      title={Understanding normalization in contrastive representation learning and out-of-distribution detection}, 
      author={Tai Le-Gia and Jaehyun Ahn},
      year={2023},
      eprint={2312.15288},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```