# DDNeRF: Depth Distribution Neural Radiance Fields
This is the official repo for the paper: Depth Distribution Neural Radiance Fields.
This repo is also contains torch implementation to MipNeRF model. 

Some functions in this repo were taken from:
#####   - https://github.com/Fyusion/LLFF
#####   - https://github.com/yenchenlin/nerf-pytorch
#####   - https://github.com/google/mipnerf

### Create and activate conda environment:
    conda env create -f environment.yml
    conda activate nerf
    
###Train model:
     
     python train_model.py --config configs/XXXX.yaml

### Eval model:
    python eval_nerf.py --logdir path_to_model_logdir
    
### Video rendering:
    python render_video.py --logdir path_to_model_logdir    
   
###example configs:
##### 1) config_360.yml - for real world 360 bounded scene using ddnerf model
##### 2) config_ff.yml - for real world forward facing scenes using ddnerf model
##### 3) config_blender.yml - for synthetic scenes using ddnerf model
##### 4) config_XXX_mipnerf.yml - for XXX scenes using mipnerf model

### Data:
##### 1) for real world data - the code support data format after colmap sparse reconstruction.
##### 2) synthetic data format is similar to NeRF blender dataset
##### 3) example real world data (motorbike scene) can be downloaded from: TBD
    
     
        
    
     
 