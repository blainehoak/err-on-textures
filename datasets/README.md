# Download the datasets:

## Prompted Textures Dataset (PTD):

The Prompted Textures Dataset (PTD) can be downloaded [here](https://zenodo.org/records/14199831) and placed inside the datasets folder under the name `ptd`.

## ImageNet-A dataset:

Natural adversarial examples from the ImageNet-A dataset can be downloaded [here](https://github.com/hendrycks/natural-adv-examples) and placed inside the datasets folder under the name `imagenet-a`.

## ImageNet (validation) data:
Download from Kaggle [here](https://www.kaggle.com/datasets/titericz/imagenet1k-val) and place inside the datasets folder under the name `imagenet-val`. If data is downloaded from a different source, just ensure it has the following directory structure:

imagenet-val/
│
├── n01440764                     # Folders named by wordnet ID
│   ├── ILSVRC2012_val_00000293.JPEG                 
│   ├── ILSVRC2012_val_00002138.JPEG            
│   ...           
├── n01443537                     # Folders named by wordnet ID
│   ├── ILSVRC2012_val_00000236.JPEG                 
│   ├── ILSVRC2012_val_00000262.JPEG            
│   ...   
| ...
├── dataset-metadata.json

### Changes to imagenet class index:

The imagenet_class_index.json has been modified such that the crane, maillot, and cardigan class duplicates have been fixed. Class 134 and 517 were both "crane" and now they are crane_bird and crane_machine respectively. Class 638 and 639 were both "maillot" but now class 638 is maillot and 639 is tank_suit. Class 264 and 474 were both "cardigan" and now they are "Cardigan_corgi" and "cardigan" respectively.