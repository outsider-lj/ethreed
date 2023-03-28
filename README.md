# ET-HREED

This repository is the implementation of paper：*Generating Empathetic Responses through Emotion Tracking and Constraint Guidance*

## empathetic dialogue generation model ethreed

### Dependencies

Python 3.6 
torch-cuda 1.7

* Download  [**Pretrained GloVe Embeddings**](http://nlp.stanford.edu/data/glove.6B.zip) and save it in `/vectors`.

* The preprocessed dataset is saved as `/datasets/ED_add_lables.json`.(https://drive.google.com/drive/folders/1DiLkM2vtDkCuvh2rcpUnPT-NV2NkkNUL) If you want to create the dataset yourself, you can run 'lj_EDjson_preprocess.py' and 'add_labels/add_labels.py'. The preprocessed dataset would be generated after the training script.

### Training

run lj_train.py

### Testing

run lj_test.py

