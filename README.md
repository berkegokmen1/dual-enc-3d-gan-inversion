# Dual Encoder GAN Inversion for High-Fidelity 3D Head Reconstruction from Single Images
### Accepted to NeurIPS 2024.

[Paper](https://arxiv.org/abs/2409.20530) | [Project Website](https://berkegokmen1.github.io/dual-enc-3d-gan-inv/) | [BibTeX](#bibtex)

## Authors
[Ahmet Berke Gökmen](https://www.linkedin.com/in/berkegokmen/)\*, [Bahri Batuhan Bilecen](https://three-bee.github.io/)\*, [Ayşegül Dündar](https://www.cs.bilkent.edu.tr/~adundar/)
<br>\* Indicates Equal Contribution


## Results

https://github.com/user-attachments/assets/3b2524f5-252b-4c27-bbb5-6f1796fa2e63

## TODO
- [X] Release Website
- [X] Release Paper
- [X] Release Code
- [X] Release Checkpoints

## Requirements and installation
* Make sure you have 64-bit Python 3.8, PyTorch 11.1 (or above), and CUDA 11.3 (or above).
* Preferably, create a new environment via [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [venv](https://docs.python.org/3/library/venv.html) and activate the environment.
* Clone repository: ```git clone --recursive https://github.com/berkegokmen1/dual-enc-3d-gan-inversion```
* Install pip dependencies: ```cd ./dual-enc-3d-gan-inversion && pip install -r requirements.txt```

## Checkpoints
| **Network** | **Filename** |
|:-----------:|:------------:|
| [PanoHead](https://drive.google.com/drive/folders/1m517-F1NCTGA159dePs5R5qj02svtX1_) | `easy-khair-180-gpc0.8-trans10-025000.pkl` |
| [Latent Avg.](https://drive.google.com/drive/folders/1LC_VSsvVKZyZWDLSsgwsWvVt1FAWDBrT?usp=share_link) | `latent_avg.npy` |
| [Visible Net](https://drive.google.com/drive/folders/1LC_VSsvVKZyZWDLSsgwsWvVt1FAWDBrT?usp=share_link) | `visible_net.pt` |
| [Occluded Net](https://drive.google.com/drive/folders/1LC_VSsvVKZyZWDLSsgwsWvVt1FAWDBrT?usp=share_link) | `occluded_net.pt` |
| [IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view) | `model_ir_se50.pth` |
| [CurricularFace Backbone](https://drive.google.com/file/d/1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj/view) | `CurricularFace_Backbone.pth` |
| [MTCNN](https://drive.google.com/file/d/1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja/view) | `mtcnn/` |

> Make sure to update `configs/paths_config.py` accordingly.


## Dataset preparation
1- Download [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset) and [LPFF Dataset](https://github.com/oneThousand1000/LPFF-dataset). Combine two datasets using LPFF's approach.

2- Then follow [PanoHead's approach](https://github.com/SizheAn/PanoHead/blob/main/3DDFA_V2_cropping/cropping_guide.md) for pose extraction and face alignment.  For this, you need to follow the setup procedure of PanoHead and ensure that you **do not skip** the setup of [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2).

3- On the combined dataset, run background removal tool using the following command: `./remove-background.sh` by setting appropriate paths.

4- Lastly, run `./gen_synth_data.sh` to create the synthetic dataset detailed in the paper. Make sure to set appropriate paths.

## Training
You can run the following commands to train two encoders separately.

```sh
./train_occluded.sh
./train_visible.sh
```

## Inference
You can run the following command to infer the trained checkpoints or use the downloaded ones and generate images, videos and meshes.

```sh
./infer.sh
```

## Contributions

Pull requests are welcome.

## Questions

You may reach me through [LinkedIn](https://www.linkedin.com/in/berkegokmen/).

## BibTeX
```
@misc{bilecen2024dualencoderganinversion,
      title={Dual Encoder GAN Inversion for High-Fidelity 3D Head Reconstruction from Single Images}, 
      author={Bahri Batuhan Bilecen and Ahmet Berke Gokmen and Aysegul Dundar},
      year={2024},
      eprint={2409.20530},
      archivePrefix={NeurIPS},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.20530}, 
}
```

## License

Copyright 2024 Bilkent DLR

Licensed under the Apache License, Version 2.0 (the "License")

<hr>

<div align="center">
  <img src="https://profile-counter.glitch.me/dual-enc-3d-gan-inversion/count.svg"  />
</div>
