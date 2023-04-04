# GTNet 
### **GTNet: Guided Transformer Network for Detecting Human-Object Interactions**
 [A S M Iftekhar](https://sites.google.com/view/asmiftekhar/home), Satish Kumar, R. Austin McEver, Suya You, B.S. Manjunath.
 
[Paper](https://arxiv.org/pdf/2108.00596.pdf).

GTNet got accepted to [Pattern Recognition
and Tracking XXXIV at SPIE commerce+ defence Program](https://spie.org/DCS23/conferencedetails/optical-pattern-recognition?enableBackToBrowse=true).

This codebase only contains code for vcoco dataset.

## Our Results on V-COCO dataset

|Method| mAP (Scenario 1)|
|:---:|:---:|
|[VSGNet](openaccess.thecvf.com/content_CVPR_2020/papers/Ulutan_VSGNet_Spatial_Attention_Network_for_Detecting_Human_Object_Interactions_Using_CVPR_2020_paper.pdf)| 51.8|
|[ConsNet](https://arxiv.org/abs/2008.06254)| 53.2|
|[IDN](https://arxiv.org/abs/2010.16219)| 53.3 |
|[OSGNet](https://www.tandfonline.com/doi/full/10.1080/0952813X.2020.1818293)| 53.4  |
|[Sun et al.](https://dl.acm.org/doi/10.1145/3512527.3531438)| 55.2 |
|[**GTNet**](https://arxiv.org/abs/2003.05541)| **58.3** |

## Installation & Setup
1. Clone repository (recursively):
```Shell
git clone --recursive https://github.com/UCSB-VRL/GTNet.git
cd GTNet
```
2. Please find the data,annotations,object detection results and embeddings for vcoco [here](https://drive.google.com/drive/folders/1RTPhhGWy0tyrO1mx6qAKjKyLfEwZqI23?usp=share_link). Download it, unzip and setup the path to directory by running:
```
python3 setup.py -d <full path to the downloaded folder>
```
Folder description can be found in our old [work](https://github.com/ASMIftekhar/VSGNet)

3. Setup enviroment by running (used python 3.6.9):
```
pip3 install -r requirements.txt
```
4. Download the best model from [here](https://drive.google.com/file/d/1cm9ICBSJZK3OuMWoxF2rgznDM4Vf-V7s/view?usp=sharing) and keep it inside a folder in the repository. We assume that you put it inside soa_vcoco folder in the repository. You can change it to anything you want.

## Inference & Training
All commands need to be run from the scripts folder.

To dump results from the best model:
```
bash run_inference.sh
```
Be sure to keep the downloaded best model in soa_vcoco folder in the repository, if you put it some other places, change the bash file accordingly.
After that, to get the results in the paper run:
```
bash run_eval_vcoco.sh
```
 
To train with 8 GPUS run:
```
bash run_train.sh
```
Please check [main.py](https://github.com/UCSB-VRL/GTNet/blob/master/scripts/main.py) for various flags.

Please contact Iftekhar (iftekhar@ucsb.edu) for any queries.




