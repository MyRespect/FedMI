### FedMI

Supplemental Material for VLDB Submission

#### Running Environment

The code has been evaluated on a deep learning stack server, which is equipped with Intel(R) Xeon(R) Gold 5218R 2.10GHz and 8 RTX A6000 GPU cards. Python 3.8.10, Pytorch 1.10.1+cu111, sklearn 1.3.2, numpy 1.22.0

We use the federated learning framework Flower for implementing the FedMI and baselines: local training, global training, multifedavg, fedmiwav, and autofed.

#### Code Structure

* encoder_models.py includes different encoder models corresponding to different data modalities, model training, and testing functions.
* client.py is the client deployment file for model training.
* main_server.py, server_agg_fedfuse.py, and server_agg_uniFL.py are the server deployment files for model training.
* dataset_loader.py includes functions loading datasets.
* main_unimodal.py, main_fusion_2modal.py, and main_fusion_3modal.py simulate different modality missing issues.
* local_contrastive_train.py implements local multimodal representation learning model training.


Please understand the full code and instructions will be well-organized upon the publication of the work.