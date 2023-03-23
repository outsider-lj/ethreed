#ET-HREED

This repository is the implementation of paper：*Generating Empathetic Responses through Emotion Tracking and Constraint Guidance*

##empathetic dialogue generation model ethreed

###Dependencies

Python 3.6 
torch-cuda 1.7

* Download  [**Pretrained GloVe Embeddings**](http://nlp.stanford.edu/data/glove.6B.zip) and save it in `/vectors`.

* The preprocessed dataset is saved as `/datasets/ED_add_lables.josn`. If you want to create the dataset yourself, you can run 'lj_EDjson_preprocess.py' and 'add_labels/add_labels.py'. The preprocessed dataset would be generated after the training script.

### Training

run lj_train.py

### Testing

run lj_test.py

### Others

There are some code about dialogue behavior in this code, because we want add behavior state to ET-HREED. But we dont complete this thought, so these codes are useless currently.
