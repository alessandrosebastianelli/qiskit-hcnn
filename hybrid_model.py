import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.quantum_info import SparsePauliOp


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits  = n_qubits
        self.qnn       = self._build_qnn()
        self.connector = TorchConnector(self.qnn)

    def _build_qnn(self) -> EstimatorQNN:
        feature_map = ZZFeatureMap(self.n_qubits)
        ansatz      = RealAmplitudes(self.n_qubits)
        qc          = QuantumCircuit(self.n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        input_params  = list(feature_map.parameters)
        weight_params = list(ansatz.parameters)

        # Create one Pauli Z observable per qubit (⟨Zᵢ⟩)
        observables = [
            SparsePauliOp.from_list([(f"{'I'*i}Z{'I'*(self.n_qubits-i-1)}", 1.0)])
            for i in range(self.n_qubits)
        ]

        return EstimatorQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=weight_params,
            observables=observables,  # <- vector output of ⟨Zᵢ⟩
            input_gradients=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.connector(x)

# Hybrid AlexNet Model
class HybridAlexNet(nn.Module):
    def __init__(self, n_qubits: int = 4, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(n_qubits)
        self.quantum = QuantumLayer(n_qubits)
        self.fc2 = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.quantum(x)
        x = self.fc2(x)
        return x

# PyTorch Lightning Module
class LightningAlexNetModule(pl.LightningModule):
    def __init__(self, n_qubits: int = 4, num_classes: int = 10, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = HybridAlexNet(n_qubits=n_qubits, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

