""" model.py """

from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool as gap
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

class GCN(torch.nn.Module):
    """
    Graph-Convolutional Neural Networks
    """
    
    def __init__(self, n_features: int, hidden_channels: int, dropout: float = 0.2) -> None:
        super(GCN, self).__init__()
        self.conv_one = GCNConv(n_features, hidden_channels)
        self.conv_two = GCNConv(hidden_channels, int(hidden_channels/2))
        self.conv_three = GCNConv(int(hidden_channels/2), int(hidden_channels/4))
        self.linear = Linear(int(hidden_channels/4), 1)
        self.dropout = dropout

    def forward(self, data, edge_index, batch) -> Tuple:
        datapoints, targets = data.x, data.y
        # 1. Obtain the node embeddings
        datapoints = self.conv_one(datapoints, edge_index)
        datapoints = datapoints.relu()
        datapoints = self.conv_two(datapoints, edge_index)
        datapoints = datapoints.relu()
        datapoints = self.conv_three(datapoints, edge_index)
        
        # 2. Aggregating message passing/embedding
        datapoints = gap(datapoints, batch)
        
        # 3. Apply the final classifier
        datapoints = F.dropout()
        
        # 4. Model output from forward and loss
        output = self.linear(datapoints)
        loss: float = torch.nn.BCEWithLogitsLoss()\
                        (output, targets.reshape(-1, 1).type_as(output))
        output = torch.sigmoid(output) # Converting output probability in range [0, 1]
        
        return output, loss


class GAT(torch.nn.Module):
    """
    Graph Attention Network
    """
    
    def __init__(self, 
                 n_features: int, 
                 hidden_channels: int, 
                 heads: int = 3, 
                 dropouts: float = 0.4
                ):
        super(GAT, self).__init__()
        self.conv_one = GATConv(n_features, hidden_channels, heads=heads, dropout=dropouts)
        self.conv_two = GATConv(hidden_channels*heads, int(hidden_channels/2), heads=heads, dropout=dropouts)
        self.conv_three = GATConv(int(hidden_channels/2)*heads, int(hidden_channels/4), heads=heads, dropout=dropouts)
        self.linear = Linear(int(hidden_channels/4)*heads, 1)
        self.dropout = dropouts
    
    def forward(self, data, edge_index, batch) -> Tuple:
        datapoints, targets = data.x, data.y
        
        # 1. Obtain the node embeddings
        datapoints = self.conv_one(datapoints, edge_index)
        datapoints = datapoints.elu()
        datapoints = F.dropout(datapoints, p=self.dropout, training=self.training)
        datapoints = self.conv_two(datapoints, edge_index)
        datapoints = datapoints.elu()
        datapoints = F.dropout(datapoints, p=self.dropout, training=self.training)
        datapoints = self.conv_three(datapoints, edge_index)
        
        # 2. Aggregating message passing / embedding
        datapoints = gap(datapoints, batch)
        
        # 3. Apply the final classifier
        datapoints = datapoints.relu()
        datapoints = F.dropout(datapoints, p=self.dropout, training=self.training)
        output = self.linear(datapoints)
        
        loss: float = torch.nn.BCEWithLogitsLoss()(output, targets.reshape(-1, 1).type_as(output))
        output = torch.sigmoid(output)
        output = torch.sigmoid(output)
        
        return output, loss


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 valid_loader
                ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
    
    # training model
    def train_one_epoch(self, epoch: int) -> Tuple:
        # set model on training mode
        self.model.train()
        
        true_targets: List[float] = []
        predicted_targets: List[float] = []
        losses: List[float] = []
        
        tqdm_iter = tqdm(self.train_loader, total=len(self.train_loader))
        
        for _, datum in enumerate(tqdm_iter):
            tqdm_iter.set_description(f"Epoch {epoch}")
            self.optimizer.zero_grad()
            output, loss = self.model(datum, datum.edge_index, datum.bacth)
            targets = datum.y
            loss.backward()
            self.optimizer.step()
            
            y_true = self.process_output(targets) # for one batch
            y_proba = self.process_output(output.flatten()) # for one batch
            
            auc: float = roc_auc_score(y_true, y_proba)
            # continuous loss/auc update
            tqdm_iter.set_postfix(
                train_loss=round(loss.item(), 2),
                train_auc=round(auc, 2),
                valid_loss=None, valid_auc=None
            )
            
            losses.append(loss.item())
            true_targets.extend(list(y_true))
            predicted_targets.extend(list(y_proba))
        
        epoch_auc = roc_auc_score(true_targets, predicted_targets)
        epoch_loss = sum(losses) / len(losses)
        
        return epoch_loss, epoch_auc, tqdm_iter

    def process_output(self, output):
        output = output.cpu().detach().numpy()
        return output
    
    def validate_one_cpoch(self, progress) -> Tuple:
        progress_tracker = progress["tracker"]
        train_loss = progress["loss"]
        train_auc = progress["auc"]
        
        # model in eval mode
        self.model.eval()
        
        true_targets: List = []
        predicted_targets: List = []
        losses: List = []
        
        for _, datum in enumerate(self.valid_loader):
            outputs, loss = self.model(datum, datum.edge_index, datum.batch)
            outputs, targets = outputs.flatten(), datum.y
            
            y_proba = self.process_outputs(outputs) # for one batch
            y_true = self.process_outputs(targets) # for one batch
            
            true_targets.extend(list(y_true))
            predicted_targets.extend(list(y_proba))
            losses.append(loss.item())
        
        epoch_auc = roc_auc_score(true_targets, predicted_targets)
        epoch_loss = sum(losses) / len(losses)
        
        progress_tracker.set_postfix(
            train_loss=round(train_loss, 2),
            train_auc=round(train_auc, 2),
            valid_loss=round(epoch_loss, 2),
            valid_auc=round(epoch_auc, 2)
        )
        
        progress_tracker.close()
        return epoch_auc, epoch_auc

    def run(self, n_epochs: int = 10):
        train_scores: List = []
        train_losses: List = []
        valid_scores: List = []
        valid_losses: List = []
        
        for epoch in range(1, n_epochs+1):
            epoch_loss, epoch_auc, progress_tracker =\
                self.train_one_epoch(epoch=epoch)

            train_losses.append(epoch_loss)
            train_scores.append(epoch_auc)
        
            # validate this epoch
            progress: Dict[str, float] = {
                "tracker": progress_tracker, 
                "loss": epoch_loss,
                "auc": epoch_auc
                }

            valid_loss, valid_auc = self.validate_one_cpoch(progress=progress)
            valid_losses.append(valid_loss)
            valid_scores.append(valid_auc)
        
        return (train_losses, train_scores), (valid_losses, valid_scores)
    
    def predict(self, test_loader) -> np.array:
        # set model on evaluation mode
        self.model.eval()
        predictions: List = []
        tqdm_iter = tqdm(test_loader, total=len(test_loader))
        
        for _, datum in tqdm_iter:
            tqdm_iter.set_description(f"Making predictions")
            with torch.no_grad():
                output, _ = self.model(datum, datum.edge_index, datum.batch)
                output = self.process_output(output.flatten())
                predictions.extend(list(output))
            
            tqdm_iter.set_postfix(stage="test dataloader")
            tqdm_iter.close()
        
        return np.array(predictions)