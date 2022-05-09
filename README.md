## Environment
Make sure your local environment has the following installed:

    Python3.9
    torch == 1.9.0
    wandb == 0.9.7
    

Install the dependency using:

    pip install -r requirements.txt


## Download data

Download data from [here](http://snap.stanford.edu/betae/KG_data.zip) and put it under `data` folder.

The directory structure should be like `PROJECT_DIR/data/NELL-betae/train-queries.pkl`, etc.


Only FB15k-237 and NELL995 are used in our study.


## Train
Training script example: `./run.sh`

It usually takes 4 days to a week to finish a run on  NVIDIA® GP102 TITAN Xp (12GB) GPU. 


## Test

The trained model will be automatically stored under `./trained_models`. The model name will be `[WANDB_RUN_NAME].pt`.

To test a trained model, you can use the following command:

    python ./test-pretrained-model.py [DATA_NAME] [WANDB_RUN_NAME]


### Test the pretrained model

The pretrained model for NELL can be downloaded here. You can use the following command to test it:

    python ./test-pretrained-model.py NELL feasible-resonance-1518


