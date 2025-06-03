
<details open>
<summary>⚠️ Work in Progress / Experimental Code</summary>
<br>

This repository contains code that is currently in the implementation and development phase. The features and functionalities are not fully tested or production-ready.

Use this code at your own risk. It may contain bugs, incomplete implementations, or performance issues. The APIs and interfaces are subject to change without notice as the project evolves.

Contributions, feedback, and bug reports are highly appreciated to help improve the codebase. Please verify and test carefully before applying it to critical or real-world projects.


</details>

# QML4EO-tutorial: Hybrid Quantum Convolutional Neural Network Classifier

## This is a qiskit adapted version of https://github.com/alessandrosebastianelli/QML4EO-tutorial/tree/main


You can run this tutorial on colab with **Runtime>Change runtme type> T4 GPU**:


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alessandrosebastianelli/qiskit-hcnn/blob/main/HQCNN.ipynb)


In any case you can run everything on your hardare, but keep attention to the enviroment (so skip cells that want to install packages) (follow the [requirements file](requirements.txt)). 

## Main references
- **Sebastianelli, A., Del Rosso, M. P., Ullo, S. L., & Gamba, P. (2023). On Quantum Hyperparameters Selection in Hybrid Classifiers for Earth Observation Data.**
- **Sebastianelli, A., Zaidenberg, D. A., Spiller, D., Le Saux, B., & Ullo, S. L. (2021). On circuit-based hybrid quantum neural networks for remote sensing imagery classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15, 565-580.**
- **Zaidenberg, D. A., Sebastianelli, A., Spiller, D., Le Saux, B., & Ullo, S. L. (2021, July). Advantages and bottlenecks of quantum machine learning for remote sensing. In 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS (pp. 5680-5683). IEEE.**
- Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
- Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.
- https://qiskit.org/documentation/machine-learning/tutorials/index.html
- https://pennylane.ai/qml/demos_qml.html




## For local usage
### Environment

```
conda create -n qis python=3.10
conda activate qis
conda install pip
pip install -r requirements.txt
```

### Download dataset

```
bash dowload_data.sh
```

### Train model

```
python main.py
```

### Monitor training

```
tensorboard --logdir lightning_logs
```
