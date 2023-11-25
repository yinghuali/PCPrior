# PCPrior is a test prioritization approach specifically for 3D Point Clouds.

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
    ----------------------
    ablation_study.py: script for ablation study.
    config.py: configuration script.
    data_processing.py: script for obtaining the ModelNet structured dataset.
    data_processing_s3dis.py: script for obtaining the S3DIS structured dataset.
    data_processing_shapenet50.py: script for obtaining the ShapeNet structured dataset.
    feature_extraction.py: script for obtaining space fearure and uncertainty feature.
