# qiskit-hcnn

⚠️ Work in Progress / Experimental Code

This repository contains code that is currently in the implementation and development phase. The features and functionalities are not fully tested or production-ready.

Use this code at your own risk. It may contain bugs, incomplete implementations, or performance issues. The APIs and interfaces are subject to change without notice as the project evolves.

Contributions, feedback, and bug reports are highly appreciated to help improve the codebase. Please verify and test carefully before applying it to critical or real-world projects.

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
