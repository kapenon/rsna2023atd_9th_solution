# entrypoints
Please navigate to each directory and execute with `python *.py`.

| dir | filename | description |
| --- | --- | --- |
| 0_preprocess/ | 0-1_prepare_png.py | A script to load DICOM files and perform basic preprocessing steps such as windowing and bit processing. The implementation comes from [this discussion](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427427) |
|  | 0-2_split_fold.py | A script for performing fold splitting for cross-validation in the context of computer vision using MultilabelStratifiedKFold. |
|  | 0-3_create_slice_df.py | A script that merges multiple CSV files, incorporates basic preprocessing. |
| 1_3Dsegmentation/ | 1-1_train.py | A script to perform organ segmentation using a 3D CNN. Used the implementation provided in this [solution](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607). This script is for training. |
|  | 1-2_pred.py | Perform inference using the model created in '1-1_train.py'. |
|  | 1-3_to_voxel_df.py | To efficiently crop using voxel data, create a DataFrame containing bounding box information. |
| 2_liver_spleen_kidney/ | 2-1_train.py | Training a 2.5D model to predict liver, spleen, and kidney involves using segmentation model predictions during input data preparation. |
| 3_bowel/ | 3-1-1_create_image_level_data.py | Creating a dataset for building the 'bowel' class classification model. |
|  | 3-1-2_train_image_level.py | This is the 1st stage. Since we have image-level labels, we intend to utilize them. In the 2nd stage, we'll use this as pre-training for building a patient-level model. |
|  | 3-2_train_patient.py | Use the weights obtained in the 1st stage to gather features and perform patient-level classification. |
| 4_extravasation/ | 4-1_train_image_level.py | This is the 1st stage for extravasation detection. We will perform binary classification using a 2D CNN and the 'image level.csv' data. |
|  | 4-2_train_patient.py | Similar to the bowel model, we will aggregate features from the 1st stage and conduct patient-level learning. |
