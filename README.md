# MemSAC

MemSAC: Memory Augmented Sample Consistency for Large-Scale Domain Adaptation 

Supplementary Material - Code and Models
Paper Id: 7167

## The following dependencies are required.

- Ubuntu 18.04
- Python==3.7.4
- numpy==1.19.2
- PyTorch==1.4.0, torchvision==0.6.0, cudatoolkit==10.1

## Training model on DomainNet

To train the model on DomainNet, run the following script.
bash jobs/domainNet_345.sh <source> <target> <Path for DomainNet dataset>

To train the model on CUB-Drawings, run the following script.
bash jobs/cub_drawings.sh <source> <target> <Path for cub2011 dataset>

## Testing using trained model.

The trained models for MemSAC are available for download at the following links:

drawing -> cub : https://drive.google.com/file/d/1ngeSrrxgzhgRt4V-zpX4mRMGnRwKKIM2/view?usp=sharing
real -> clipart : https://drive.google.com/file/d/1rtzZDwnCZJm3oz7dvb2hWukUPHEb4tCZ/view?usp=sharing
painting -> real : https://drive.google.com/file/d/1qDr5TEAWGujBMY9yr8jSAKDxRVx-bt1r/view?usp=sharing

To directly test our trained model, download these models and use the following commands.

*Drawings -> CUB*
python eval.py --gpu_id 0 --nClasses 200 --checkpoint drawing_cub.pth.tar --data_dir <Path for cub2011 dataset> --batch_size 64 --dataset cub2011 --target cub

*real -> clipart*
python eval.py --gpu_id 0 --nClasses 345 --checkpoint real_clipart.pth.tar --data_dir <Path for domainNet dataset>  --batch_size 64 --dataset domainNet_full --target clipart

*painting -> real*
python eval.py --gpu_id 0 --nClasses 345 --checkpoint painting_real.pth.tar --data_dir <Path for domainNet dataset>  --batch_size 64 --dataset domainNet_full--target real

Paper: MemSAC: Memory Augmented Sample Consistency for Large-Scale Domain Adaptation
Supplementary material
code and models
Paper Id: 7167