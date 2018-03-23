# [Single-Shot Object Detection with Enriched Semantics](https://arxiv.org/abs/1712.00433)
---

### Results on VOC

| System | VOC2007 test *mAP* | VOC2012 test *mAP* | VOC2007 test *mAP* trained with COCO | VOC2012 test *mAP* trained with COCO
|:-------|:-----:|:-------:|:-------:|:-------:|
| SSD300* (VGG16) | 77.2 | 75.8 | 81.2 | 79.3 |
| SSD512* (VGG16) | 79.8 | 78.5 | 83.2 | 82.2 |
| DES300 (VGG16) | 79.7 | [77.1](http://host.robots.ox.ac.uk:8080/anonymous/RCMS6B.html) | 82.7 | [81.0](http://host.robots.ox.ac.uk:8080/anonymous/IRJJ5L.html) |
| DES512 (VGG16) | 81.7 | [80.3](http://host.robots.ox.ac.uk:8080/anonymous/OBE3UF.html) | 84.3 | [83.7](http://host.robots.ox.ac.uk:8080/anonymous/MURP2C.html) |

### Results on COCO
| System | 0.5:0.95 | 0.5 | 0.75
|:-------|:-----:|:-------:|:-------:|
| SSD300* (VGG16) | 25.1 | 43.1 | 25.8 |
| SSD512* (VGG16) | 28.8 | 48.5 | 30.3 |
| DES300 (VGG16) | 28.3 | 47.3 | 29.4 |
| DES512 (VGG16) | 32.8 | 53.2 | 34.6 |

### Citing DES

Please cite DES in your publications if it helps your research:

    @inproceedings{zhang2018single,
      title = {Single-Shot Object Detection with Enriched Semantics},
      author = {Zhang, Zhishuai and Qiao, Siyuan and Xie, Cihang and Shen, Wei and Wang, Bo and Yuille, Alan L.},
      booktitle = {CVPR},
      year = {2018}
    }

### Trained Models
1. VOC07
* [DES300x300](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/Ec2soyecYTNPsGeRgfg8AEoB_MpJT-LStQ74v55UQmpnGw?e=udCX3C)
* [DES512x512](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EdNxwuQKz91PssS5Ca2aUpwBtOSrIIXIDDPwRwnvwWYK8g?e=TKNEbn)
* [DES300x300_COCO](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EdnSKR40d7BKoayL1qx8ea4BNh8Th4XYmU2s4fgu2_B96A?e=tXcpya)
* [DES512x512_COCO](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EcOOm2DQ1ARJvU7LYmqUvPIBFsaQgc_sMC3_lpPdDjUWOw?e=PpnaRX)
2. VOC12
* [DES300x300](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EdWEjdw76dBHrNGTXvH0inIBKiSOQBLkLGRumGzWfGHzXQ?e=3stbWp)
* [DES512x512](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EYTJi3NuaCFFgyBWXx_NVw0BDAuYsLCmp2SiolbQHC-Mlg?e=DHHyDD)
* [DES300x300_COCO](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EVPQxE-Hd-RJuBUQG8y9dEsB9Iwof5JZTEi5YOv8cTUeYA?e=tFz6s9)
* [DES512x512_COCO](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EadX_3ukbJlKkuLRbZTpFrYBF6PEkc0wSmFiirCwEJS4gw?e=LChnXA)
3. COCO
* [DES300x300](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/EXp3iBOV4B5Fgp_DmCrSF94BLvYez7OUmd2U7jk36_Ea_A?e=ya8OiV)
* [DES512x512](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/zzhang99_jh_edu/ES9ALrORx4VFkjs6DjKg6RkBk5Z2PH229haJAnkXCExZJg?e=jWm0lN)

### Installation and Preparation
1. Clone the code.

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

3. Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in `$HOME/data/`
  ```Shell
  # Download the data.
  cd $HOME/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # Extract the data.
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```

4. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
  ./data/VOC0712/create_list.sh
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
  # and make soft links at examples/VOC0712/
  ./data/VOC0712/create_data.sh
  ```

### Train/Eval
All `.sh` files in jobs folder are for training or evaluation. For training, use the models in `initial_models` to initialize weights. For evaluation, use the model links above to initialize weights.
