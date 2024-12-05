# Full-body Virtual Try-On using Top and Bottom Garments with Wearing Style Control
Official implementation for "Full-body Virtual Try-On using Top and Bottom Garments with Wearing Style Control" published in Computer Vision and Image Understanding.
The code and pre-trained models are tested with pytorch 1.31.1, torchvision 0.13.1, Python 3.8, CUDA 11.6.

![Teaser](./fig_WGVITONresult3x3.png)

## Usage
WGF-VITON is a single stage network to synthesize VITON image simultanously using top and bottom clothes with wearing style control.

## Dataset : Fashion-TB
![Teaser](./data_teaser.png)

## Installation (Anaconda)
```
conda create -n {name} python=3.8 anaconda
conda activate {name}
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install cupy
```

## Installation (Docker)
The folder "docker" has Dockerfile to set docker images for running WGF-VITON.
```
cd docker
docker build . -t {docker_image_name}
```

For the pre-trained model, you can donwload weights from WGF-VITON.

### Training

```python
python train.py --name {project_name} --gpu_ids 0,1 --dataroot {data_path} --keep_step 50000 --decay_step 150000 -b 4
```

### Test

```python
python test.py --name {project_name} --gpu_ids 0,1 --dataroot {data_path} --wearing {test json file} -b 8 --checkpoint {checkpoint_path}
```

## License

All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.

## Citation
```
@article{park2022single,
  title={Full-body Virtual Try-on using Top and Bottom Garments with Wearing Style Control},
  author={Park, Soonchan and Park, Jinah},
  journal={Computer Vision and Image Understanding},
  year={2024}
}
```

## Acknoledgements
We implemente the code for WGF-VITON based on PyTorch implementation of [CP-VTON](https://github.com/sergeywong/cp-vton), [SPADE](https://github.com/NVlabs/SPADE), and [HR-VITON](https://github.com/sangyun884/HR-VITON).
