"""
Hybrid quantum-classical models, built with PennyLane's `default.qubit`
simulator (no real QPU needed; swap the device for a real backend,
e.g. qiskit's IBM runtime, by changing `qml.device(...)`).

Two primary quantum model architectures:

1. QuantumKernelSVM   - encodes classical features into quantum states
   (angle embedding), estimates a quantum kernel (state overlap fidelity)
   between samples, and hands the precomputed quantum kernel matrix to
   a classical maximum-margin SVM.
   Includes both exact PennyLane QNodes for single-circuit inspection
   and vectorized state fidelity computation for scalable training.

2. VariationalQuantumClassifier - a trainable parameterized quantum circuit
   (AngleEmbedding + StronglyEntanglingLayers) whose output expectation value
   is optimized with Adam against clinical labels.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.svm import SVC

from . import config


# ------------------------------------------------------------------ #
# 1. Quantum Kernel SVM
# ------------------------------------------------------------------ #
class QuantumKernelSVM:
    def __init__(self, n_qubits=config.N_QUBITS, c=1.0, random_state=config.RANDOM_STATE):
        self.n_qubits = n_qubits
        self.c = c
        self.random_state = random_state
        self.svc = SVC(C=c, kernel="precomputed", probability=True, random_state=random_state)
        self._X_train = None
        self._build_circuit()

    def _build_circuit(self):
        """(Re)builds the PennyLane device/qnode. Called from __init__ and
        from __setstate__, because a qnode closure cannot be pickled directly.
        We save the picklable configuration and reconstruct the circuit upon joblib.load()."""
        n_qubits = self.n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def _kernel_circuit(x1, x2):
            qml.AngleEmbedding(x1, wires=range(n_qubits))
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        self._kernel_circuit = _kernel_circuit

    def evaluate_qnode(self, x1, x2):
        """Executes the PennyLane QNode directly for a pair of feature vectors.
        Returns the ground state measurement probability |<psi(x1)|psi(x2)>|^2."""
        return float(self._kernel_circuit(x1, x2)[0])

    def __getstate__(self):
        return {
            "n_qubits": self.n_qubits,
            "c": self.c,
            "random_state": self.random_state,
            "svc": self.svc,
            "_X_train": self._X_train,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_circuit()

    def _pairwise_kernel(self, A, B):
        """Computes the exact quantum kernel matrix K_{ij} = |<psi(a_i)|psi(b_j)>|^2.
        For angle embedding without entanglement, |<psi(a)|psi(b)>|^2 evaluates exactly to:
        prod_{k=0}^{n-1} cos^2((a_k - b_k) / 2).
        Vectorized across (len(A), len(B)) for high numerical efficiency while remaining
        100% mathematically exact to the simulated PennyLane QNode."""
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        diff = (A[:, None, :] - B[None, :, :]) / 2.0
        cos_sq = np.cos(diff) ** 2
        K = np.prod(cos_sq, axis=-1)
        return np.clip(K, 0.0, 1.0)

    def fit(self, X_train, y_train):
        self._X_train = np.asarray(X_train, dtype=np.float64)
        K_train = self._pairwise_kernel(self._X_train, self._X_train)
        self.svc.fit(K_train, y_train)
        return self

    def predict(self, X_test):
        K_test = self._pairwise_kernel(X_test, self._X_train)
        return self.svc.predict(K_test)

    def predict_proba(self, X_test):
        K_test = self._pairwise_kernel(X_test, self._X_train)
        return self.svc.predict_proba(K_test)

    def decision_function(self, X_test):
        K_test = self._pairwise_kernel(X_test, self._X_train)
        return self.svc.decision_function(K_test)


# ------------------------------------------------------------------ #
# 2. Variational Quantum Classifier (trainable circuit)
# ------------------------------------------------------------------ #
class VariationalQuantumClassifier:
    def __init__(self, n_qubits=config.N_QUBITS, n_layers=config.QUANTUM_LAYERS,
                 lr=0.08, epochs=35, batch_size=16, random_state=config.RANDOM_STATE):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self._build_circuit()
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.weights = pnp.array(self.rng.normal(0, 0.1, size=shape), requires_grad=True)
        self.bias = pnp.array(0.0, requires_grad=True)

    def _build_circuit(self):
        n_qubits = self.n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            qml.AngleEmbedding(x, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def __getstate__(self):
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "weights": np.array(self.weights),
            "bias": float(self.bias),
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rng = np.random.default_rng(self.random_state)
        self._build_circuit()
        self.weights = pnp.array(self.weights, requires_grad=False)
        self.bias = pnp.array(self.bias, requires_grad=False)

    def _predict_raw(self, weights, bias, X):
        return pnp.array([self.circuit(weights, x) + bias for x in X])

    def fit(self, X_train, y_train):
        y_pm1 = np.where(y_train == 0, -1.0, 1.0)
        opt = qml.AdamOptimizer(stepsize=self.lr)
        weights, bias = self.weights, self.bias
        n = len(X_train)

        def cost(w, b, Xb, yb):
            preds = self._predict_raw(w, b, Xb)
            return pnp.mean((preds - yb) ** 2)

        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                Xb, yb = X_train[batch_idx], y_pm1[batch_idx]
                (weights, bias), _ = opt.step_and_cost(
                    lambda w, b: cost(w, b, Xb, yb), weights, bias
                )
        self.weights, self.bias = weights, bias
        return self

    def predict_proba(self, X):
        raw = np.array(self._predict_raw(self.weights, self.bias, X))
        proba_pos = (raw + 1.0) / 2.0  # squash expectation [-1, 1] to [0, 1]
        proba_pos = np.clip(proba_pos, 0.0, 1.0)
        return np.stack([1.0 - proba_pos, proba_pos], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
