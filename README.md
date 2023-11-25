# PCPrior
PCPrior is a test prioritization approach specifically for 3D Point Clouds.

## Main Requirements
    PyTorch 2.0.0
    TensorFlow 2.5.1
    XGBoost 1.7.4
    LighGBM 3.3.5
    pytorch-tabnet 4.1.0
    scikit-learn 1.0.2

##  Repository catalogue
    example: scripts to obtain the experimental results of the paper.
    impact_pcprior: scripts of parameter analysis.
    importance: results of feature importance.
    models: scripts of 3D classification models.
    result: experimental results of the paper.
    retrain: scripts of retraining.
    target_models: the evaluated 3D classification models.
    ----------------------
    ablation_study.py: script for ablation study.
    config.py: configuration script.
    data_processing.py: script for obtaining the ModelNet structured dataset.
    data_processing_s3dis.py: script for obtaining the S3DIS structured dataset.
    data_processing_shapenet50.py: script for obtaining the ShapeNet structured dataset.
    feature_extraction.py: script for obtaining space fearure and uncertainty feature.
    feature_importance.py: script of feature importance.
    get_dgcnn_mutants_feature.py: script for obtaining mutants feaure of DGCNN model.
    get_model_dgcnn_pre.py: script for obtaining prediction feature from DGCNN model.
    get_model_pointconv_pre.py: script for obtaining prediction feature from PointConv model.
    get_model_pointnet_pre.py: script for obtaining prediction feaure of MSG, SSG and PointNet models.
    get_point_mixture.py: script for obtaining noisy datasets.
    get_point_mutants.py: script for obtaining 3D point cloud mutants datasets.
    get_point_mutants_feature.py: script for obtaining mutants feature of MSG,SSG and PointNet models.
    get_pointconv_mutants_feature.py: script for obtaining mutants feature of PointConv model.
    get_rank_idx.py: script for uncertainty approaches.
    pcprior.py: script of PCPrior for obtaining APFD.
    pcprior_dnn.py: script of PCPrior using a DNN ranking model for obtaining APFD.
    pcprior_dnn_pfd.py: script of PCPrior using a DNN ranking model for obtaining PFD.
    pcprior_pfd.py: script of PCPrior for obtaining PFD.
    pcprior_tabnet.py: script of PCPrior using the TabNet ranking model for obtaining APFD.
    pcprior_tabnet_pfd.py: script of PCPrior using the TabNet ranking model for obtaining PFD.
    provider.py: script related to models.
    train_dgcnn.py: script of traing DGCNN model.
    train_pointconv.py: script of traing PointConv model.
    train_pointnet2_msg_cls.py: script of traing MSG model.
    train_pointnet2_ssg_cls.py: script of traing SSG model.
    train_pointnet_cls.py: script of traing PointNet model.
    utils.py: script for tools.
    
## How to run PCPrior
### Step1: Download the dataset:  
https://zenodo.org/records/10206303  
This dataset includes processed 3D point clouds and extracted features.

### Step2: Run PCPrior  
```sh run_PCPrior.sh```


