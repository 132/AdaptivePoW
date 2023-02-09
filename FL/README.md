## Download miniconda env manager
https://docs.conda.io/en/latest/miniconda.html

use the requirements.txt to create an envirnment
conda create -n test_fl python=3.9.7

once created activate the env

conda activate <env> 
conda install pip
which pip 
# you should see the pip in your env

pip install -r requirements.txt

## Run as script
python3 BlockFL.py 
        args:
            -h: help
            -e: <miner/client> 
            -p: <path to shared location on BlockChain which has local models and global model> 
            -i: <Only required for client - location of where the images are> 
            -n: client name

## Run as server
python3 BlockFL.py -e miner -p <path to shared location on BlockChain which has local models and global model> 

## Run as client
python3 BlockFL.py -e <miner/client> -p <path to shared location on BlockChain which has local models and global model> -i <Only required for client - location of where the images are> 