# WG-VITON (Under review)
Official implementation for "WG-VITON: Wearing-guide Virtual Try-on for Top and Bottom Clothes".
The code and pre-trained models are tested with pytorch 1.31.1, torchvision 0.13.1, Python 3.8, CUDA 11.6.

![Teaser](./teaser.png)

## Usage
WG-VITON is a single stage network to synthesize VITON image simultanously using top and bottom clothes. 

## Installation
```
conda create -n {name} python=3.8 anaconda
conda activate {name}
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install cupy
```

## Dataset : Fashion-TB

## Training

```python
python train.py --dataroot {data_path} --keep_step 50000 --decay_step 150000 --gpu_ids 0 -b 4
```

## Test

```python
python test.py --dataroot {data_path} --wearing {test json file} --gpu_ids 0 -b 8 --checkpoint {checkpoint_path}
```

## License

All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.

## Citation
TBD.

## Acknoledgements
We implemente the code for WG-VITON based on PyTorch implementation of [CP-VTON](https://github.com/sergeywong/cp-vton), [SPADE](https://github.com/NVlabs/SPADE), and [HR-VITON](https://github.com/sangyun884/HR-VITON).
