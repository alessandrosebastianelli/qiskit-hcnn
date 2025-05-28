# qiskit-hcnn

## Environment

```
conda create -n qis python=3.10
conda activate qis
conda install pip
pip install -r requirements.txt
```

## Download dataset

```
bash dowload_data.sh
```

## Train model

```
python main.py
```

## Monitor training

```
tensorboard --logdir lightning_logs
```

