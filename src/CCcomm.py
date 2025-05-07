#CCcomm.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse

class NetworkLinearRegression(nn.Module):
    def __init__(self, n_features, alpha=0, lambda_=0.01, Omega=None, device=None):
        """
        Network Linear Regression: Linear Regression with L1 + Laplacian Regularization.
        """
        super(NetworkLinearRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # 线性层: X @ W + b
        self.alpha = alpha  # Balance coefficient between L1 and Laplacian regularization
        self.lambda_ = lambda_  # Regularization strength
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute symmetric normalized Laplacian matrix
        if Omega is not None:
            L_sym = self._compute_symmetric_laplacian(Omega)
            self.L_sym = torch.sparse_csr_tensor(
                L_sym.indptr, L_sym.indices, L_sym.data.astype(np.float32), size=L_sym.shape
            ).to(self.device)
        else:
            self.L_sym = None

    def _compute_symmetric_laplacian(self, Omega):
        D_inv_sqrt = sparse.diags(1 / np.sqrt(Omega.sum(axis=1).A1))
        I = sparse.eye(Omega.shape[0])
        L_sym = I - D_inv_sqrt @ Omega @ D_inv_sqrt
        return L_sym

    def forward(self, X):
        return self.linear(X)  

    def loss(self, predictions, y):
        """ Compute mean squared error loss and add L1 and Laplacian regularization """
        mse_loss = nn.MSELoss()(predictions, y)

        l1_penalty = self.alpha * self.lambda_ * torch.sum(torch.abs(self.linear.weight))

        if self.L_sym is not None:
            weight = self.linear.weight.squeeze()
            laplacian_penalty = (1 - self.alpha) * self.lambda_ * torch.sparse.mm(
                weight.unsqueeze(0), torch.sparse.mm(self.L_sym, weight.unsqueeze(1))
            ).squeeze()
        else:
            laplacian_penalty = torch.tensor(0.0, device=self.device)

        return mse_loss + l1_penalty + laplacian_penalty


def CCcommInfer_linear(X, y, alpha=0, lambda_=0.01, Omega=None, learning_rate=0.001, n_epochs=500):#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]

    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X, feature_means = center_features(X)
    X, y = X.to(device), y.to(device)

    model = NetworkLinearRegression(n_features, alpha, lambda_, Omega, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions = model(X)
        loss = model.loss(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

class NetworkCoxRegression(nn.Module):
    def __init__(self, n_features, alpha=0, lambda_=0.01, Omega=None, device=None):
        """
        Network Cox Regression: Cox Regression with L1 + Laplacian Regularization.

        Parameters:
        - n_features: number of input features.
        - alpha: weight for L1 vs Laplacian regularization (0 <= alpha <= 1).
        - lambda_: regularization strength.
        - Omega: similarity matrix (scipy.sparse.csr_matrix) representing the network structure.
        - device: device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super(NetworkCoxRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1, bias=False)  # Linear layer: X @ W (no bias for Cox regression)
        self.alpha = alpha                     # Balance between L1 and Laplacian regularization
        self.lambda_ = lambda_                 # Regularization strength
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute symmetric normalized Laplacian matrix
        if Omega is not None:
            L_sym = self._compute_symmetric_laplacian(Omega)  # Compute Laplacian
            self.L_sym = torch.sparse_csr_tensor(
                L_sym.indptr, L_sym.indices, L_sym.data.astype(np.float32), size=L_sym.shape
            ).to(self.device)  # Convert to PyTorch sparse tensor and move to device
        else:
            self.L_sym = None

    def _compute_symmetric_laplacian(self, Omega):
        """
        Compute the symmetric normalized Laplacian matrix from the similarity matrix Omega.
        """
        # Compute degree matrix D
        D = sparse.diags(Omega.sum(axis=1).A1)  # D is a diagonal matrix with row sums of Omega

        # Compute D^{-1/2}
        D_inv_sqrt = sparse.diags(1 / np.sqrt(Omega.sum(axis=1).A1))

        # Compute symmetric normalized Laplacian L_sym = I - D^{-1/2} Omega D^{-1/2}
        I = sparse.eye(Omega.shape[0])  # Identity matrix
        L_sym = I - D_inv_sqrt @ Omega @ D_inv_sqrt

        return L_sym

    def forward(self, X):
        """
        Forward pass: compute risk scores (log hazard ratio).
        """
        return self.linear(X)  # Output risk scores (no activation function)

    def loss(self, risk_scores, time, event):
        """
        Compute the Cox partial likelihood loss with L1 and Laplacian regularization.

        Parameters:
        - risk_scores: Model output (risk scores).
        - time: Survival times.
        - event: Event indicators (1 if event occurred, 0 if censored).
        """
        # Sort by time (ascending order)
        sorted_time, indices = torch.sort(time, descending=False)
        sorted_risk_scores = risk_scores[indices]
        sorted_event = event[indices]

        # Compute partial likelihood loss
        loss = 0.0
        for i in range(len(sorted_time)):
            if sorted_event[i] == 1:  # Only consider events (not censored)
                # Risk set: individuals still at risk at time sorted_time[i]
                risk_set = (sorted_time >= sorted_time[i]).nonzero(as_tuple=True)[0]
                # Log-sum-exp of risk scores in the risk set
                log_sum_exp = torch.logsumexp(sorted_risk_scores[risk_set], dim=0)
                # Partial likelihood contribution
                loss += sorted_risk_scores[i] - log_sum_exp

        # Negative log partial likelihood
        loss = -loss / torch.sum(event)  # Normalize by number of events

        # L1 regularization
        l1_penalty = self.alpha * self.lambda_ * torch.sum(torch.abs(self.linear.weight))

        # Laplacian regularization
        if self.L_sym is not None:
            weight = self.linear.weight.squeeze()  # Get weight vector (n_features,)
            laplacian_penalty = (1 - self.alpha) * self.lambda_ * torch.sparse.mm(
                weight.unsqueeze(0), torch.sparse.mm(self.L_sym, weight.unsqueeze(1))
            ).squeeze()  # weight^T @ L_sym @ weight
        else:
            laplacian_penalty = torch.tensor(0.0, device=self.device)

        return loss + l1_penalty + laplacian_penalty


def CCcommInfer_cox(X, time, event, alpha=0, lambda_=0.01, Omega=None, learning_rate=0.001, n_epochs=500):
    """
    Train the Network Cox Regression model.

    Parameters:
    - X: Training feature matrix (torch.Tensor or numpy.ndarray).
    - time: Survival times (torch.Tensor or numpy.ndarray).
    - event: Event indicators (torch.Tensor or numpy.ndarray).
    - alpha: Weight for L1 vs Laplacian regularization.
    - lambda_: Regularization strength.
    - Omega: Laplacian matrix (scipy.sparse.csr_matrix).
    - learning_rate: Learning rate for optimization.
    - n_epochs: Number of training epochs.

    Returns:
    - model: Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(time, torch.Tensor):
        time = torch.from_numpy(time).float()
    if not isinstance(event, torch.Tensor):
        event = torch.from_numpy(event).float()

    X, time, event = X.to(device), time.to(device), event.to(device)  # Move data to device
    model = NetworkCoxRegression(n_features, alpha, lambda_, Omega, device=device).to(device)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        # Model prediction (risk scores)
        risk_scores = model(X)
        loss = model.loss(risk_scores, time, event)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

class NetworkOrdinalLogit(nn.Module):
    def __init__(self, n_features, num_classes, alpha=0, lambda_=0.1, Omega=None, device=None):
        """
        Network Ordinal Logit Model: Ordinal Logistic Regression with L1 + Laplacian Regularization.
        
        Parameters:
        - n_features: number of input features.
        - num_classes: number of ordered classes.
        - alpha: weight for L1 vs Laplacian regularization (0 <= alpha <= 1).
        - lambda_: regularization strength.
        - Omega: similarity matrix (scipy.sparse.csr_matrix) representing the network structure.
        - device: device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super(NetworkOrdinalLogit, self).__init__()
        self.linear = nn.Linear(n_features, 1) 
        self.num_classes = num_classes
        self.alpha = alpha                     
        self.lambda_ = lambda_                 
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")      
        self.cutpoints = nn.Parameter(torch.arange(num_classes - 1).float() - num_classes / 2).to(self.device)

        # Compute symmetric normalized Laplacian matrix
        if Omega is not None:
            L_sym = self._compute_symmetric_laplacian(Omega)  # Compute Laplacian
            self.L_sym = torch.sparse_csr_tensor(
                L_sym.indptr, L_sym.indices, L_sym.data.astype(np.float32), size=L_sym.shape
            ).to(self.device) 
        else:
            self.L_sym = None

    def _compute_symmetric_laplacian(self, Omega):
        D_inv_sqrt = sparse.diags(1 / np.sqrt(Omega.sum(axis=1).A1))
        I = sparse.eye(Omega.shape[0])  
        L_sym = I - D_inv_sqrt @ Omega @ D_inv_sqrt
        
        return L_sym

    def forward(self, X):
        """
        Forward pass: compute cumulative probabilities for ordinal classes.
        """
        logits = self.linear(X)  
        sigmoids = torch.sigmoid(self.cutpoints - logits) 
        return sigmoids
    

    def loss(self, predictions, y):
        """
        Compute the negative log-likelihood loss for ordinal logit model with L1 and Laplacian regularization.
        """
        sigmoids = predictions
        sigmoids = torch.cat([torch.zeros_like(sigmoids[:, [0]]), sigmoids, torch.ones_like(sigmoids[:, [0]])], dim=1)
        
        class_probs = sigmoids[:, 1:] - sigmoids[:, :-1]

        y_true = y.long().squeeze()
        likelihoods = torch.gather(class_probs, 1, y_true.unsqueeze(1))
        nll_loss = -torch.log(likelihoods + 1e-15).mean()

        l1_penalty = self.alpha * self.lambda_ * torch.sum(torch.abs(self.linear.weight))

        if self.L_sym is not None:
            weight = self.linear.weight.squeeze()  
            laplacian_penalty = (1 - self.alpha) * self.lambda_ * torch.sparse.mm(
                weight.unsqueeze(0), torch.sparse.mm(self.L_sym, weight.unsqueeze(1))
            ).squeeze()  # weight^T @ L_sym @ weight
        else:
            laplacian_penalty = torch.tensor(0.0, device=self.device)

        return nll_loss + l1_penalty + laplacian_penalty


def CCcommInfer_ordinal_logit(X, y, alpha=0, lambda_=0.1, Omega=None, learning_rate=0.0001, n_epochs=500, device=None):
    """
    Train the Network Ordinal Logit model.
    
    Parameters:
    - X: Training feature matrix (torch.Tensor or numpy.ndarray).
    - y: Training labels (torch.Tensor or numpy.ndarray).
    - alpha: Weight for L1 vs Laplacian regularization.
    - lambda_: Regularization strength.
    - Omega: Laplacian matrix (scipy.sparse.csr_matrix).
    - learning_rate: Learning rate for optimization.
    - n_epochs: Number of training epochs.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    
    Returns:
    - model: Trained model.
    - feature_means: Means of each feature (used for centering).
    """
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]
    y = y.max() - y# Make sure y starts at 0
    num_classes = len(np.unique(y))

    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X, feature_means = center_features(X)
    X, y = X.to(device), y.to(device)  

    model = NetworkOrdinalLogit(n_features, num_classes, alpha, lambda_, Omega, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions = model(X)
        loss = model.loss(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    return model

class NetworkLogisticRegression(nn.Module):
    def __init__(self, n_features, alpha=0, lambda_=0.01, Omega=None, device=None):
        """
        Network Logistic Regression: Logistic Regression with L1 + Laplacian Regularization.

        Parameters:
        - n_features: number of input features.
        - alpha: weight for L1 vs Laplacian regularization (0 <= alpha <= 1).
        - lambda_: regularization strength.
        - Omega: similarity matrix (scipy.sparse.csr_matrix) representing the network structure.
        - device: device to run the model on (e.g., 'cuda' or 'cpu').
        """
        super(NetworkLogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # Linear layer: X @ W + b
        self.sigmoid = nn.Sigmoid()            # Activation function: Sigmoid
        self.alpha = alpha                     # Balance between L1 and Laplacian regularization
        self.lambda_ = lambda_                 # Regularization strength
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute symmetric normalized Laplacian matrix
        if Omega is not None:
            L_sym = self._compute_symmetric_laplacian(Omega)  # Compute Laplacian
            self.L_sym = torch.sparse_csr_tensor(
                L_sym.indptr, L_sym.indices, L_sym.data.astype(np.float32), size=L_sym.shape
            ).to(self.device)  
        else:
            self.L_sym = None

    def _compute_symmetric_laplacian(self, Omega):
        D_inv_sqrt = sparse.diags(1 / np.sqrt(Omega.sum(axis=1).A1))
        # Compute symmetric normalized Laplacian L_sym = I - D^{-1/2} Omega D^{-1/2}
        I = sparse.eye(Omega.shape[0])  
        L_sym = I - D_inv_sqrt @ Omega @ D_inv_sqrt

        return L_sym

    def forward(self, X):
        """
        Forward pass: compute logits and apply sigmoid.
        """
        logits = self.linear(X) 
        return self.sigmoid(logits)  

    def loss(self, predictions, y):
        """
        Compute the logistic loss with L1 and Laplacian regularization.
        """
        # Binary cross-entropy loss
        bce_loss = nn.BCELoss()(predictions, y)

        # L1 regularization
        l1_penalty = self.alpha * self.lambda_ * torch.sum(torch.abs(self.linear.weight))

        if self.L_sym is not None:
            weight = self.linear.weight.squeeze()  
            laplacian_penalty = (1 - self.alpha) * self.lambda_ * torch.sparse.mm(
                weight.unsqueeze(0), torch.sparse.mm(self.L_sym, weight.unsqueeze(1))
            ).squeeze()# weight^T @ L_sym @ weight
        else:
            laplacian_penalty = torch.tensor(0.0, device=self.device)

        return bce_loss + l1_penalty + laplacian_penalty


def CCcommInfer_logit(X, y, alpha=0, lambda_=0.01, Omega=None, learning_rate=0.0001, n_epochs=500):
    """
    Train the Network Logistic Regression model. 

    Parameters:
    - X: Training feature matrix (torch.Tensor or numpy.ndarray).
    - y: Training labels (torch.Tensor or numpy.ndarray).
    - n_features: Number of input features.
    - alpha: Weight for L1 vs Laplacian regularization.
    - lambda_: Regularization strength.
    - Omega: Laplacian matrix (scipy.sparse.csr_matrix).
    - learning_rate: Learning rate for optimization.
    - n_epochs: Number of training epochs.

    Returns:
    - model: Trained model.
    - feature_means: Means of each feature (used for centering).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()

    X, feature_means = center_features(X)
    X, y = X.to(device), y.to(device)  

    model = NetworkLogisticRegression(n_features, alpha, lambda_, Omega, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        predictions = model(X)
        loss = model.loss(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    return model

def center_features(X):
    feature_means = X.mean(dim=0, keepdim=True)
    X_centered = X - feature_means
    return X_centered, feature_means

class CCcommInfer:
    def __init__(self, method="linear", **kwargs):
        self.method = method.lower()
        self.model = self._initialize_model(**kwargs)
    
    def _initialize_model(self, **kwargs):
        if self.method == "linear":
            return CCcommInfer_linear(**kwargs)
        elif self.method == "cox":
            return CCcommInfer_cox(**kwargs)
        elif self.method == "ordinal":
            return CCcommInfer_ordinal_logit(**kwargs)
        elif self.method == "binary":
            return CCcommInfer_logit(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")