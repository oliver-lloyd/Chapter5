import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from torch.nn import CosineEmbeddingLoss
from torch.nn.functional import cosine_similarity
from time import time

from baseline_model import GCN

torch.manual_seed(12345)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using CUDA: {torch.cuda.is_available()}')


def train(model, optimizer, criterion, cosine_target=[1.0]):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(h[data.train_mask], data.y[data.train_mask], torch.tensor(cosine_target).to(device)) # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

parser = argparse.ArgumentParser()
parser.add_argument('data_object_path')
args = parser.parse_args()

# Load data to get stats
data = torch.load(args.data_object_path)
n_nodes = data.num_nodes
n_features = data.x.shape[-1]

# Load partial results if they exist
trace_path = 'LOOCV_cosines.csv'
vectors_path = 'learned_vecs.csv'
try:
    trace_df = pd.read_csv(trace_path)
    learned_vecs = pd.read_csv(vectors_path)
except FileNotFoundError:
    trace_df = pd.DataFrame(columns=['left_out_node', 'left_out_ID', 'cosine_to_actual', 'best_train_loss', 'best_epoch', 'time_to_best'])
    learned_vecs = pd.DataFrame(columns=['drug'] + [str(i) for i in range(n_features)])

# Run LOOCV
for node_id in range(n_nodes):
    if node_id in trace_df.left_out_ID:
        continue  # Skip if split already run
    else:
        print(f'Testing on node {node_id + 1}/{n_nodes}')

        # Re-instantiate graph
        data = torch.load(args.data_object_path).to(device)

        # Create nodesplit masks
        train_mask = torch.ones(data.num_nodes).to(device)
        train_mask[node_id] = 0
        train_mask = train_mask.to(bool)
        data.train_mask = train_mask
        data.test_mask = ~train_mask  # Invert bool tensor using tilde
        
        # Zero out features for test node
        data.x[data.test_mask] = torch.zeros(data.x.shape[-1]) .to(device) 
        
        # Configure model
        model = GCN(data.num_features).to(device)
        criterion = CosineEmbeddingLoss().to(device)  # Define loss criterion.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Define optimizer.
        scheduler_patience = 10
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience)
        
        # Training loop
        early_stop_threshold = scheduler_patience * 5  # Try 5 rounds of reducing learning rate
        best_loss = 1e9
        epochs_since_improved = 0
        losses = []
        start = time()
        for epoch in range(1000):

            # Calculate hidden layer and loss
            train_loss, h = train(model, optimizer, criterion)
            losses.append([epoch, train_loss.item()])
            scheduler.step(train_loss)
            
            # Early stopping checks
            if train_loss.item() < best_loss:
                time_to_best = time()-start
                best_loss = train_loss.item()
                best_h = h
                best_epoch = epoch
                epochs_since_improved = 0
            else:
                epochs_since_improved += 1

                # Stop if done
                if epochs_since_improved >= early_stop_threshold:
                    break

        # Calculate leave-one-out performance by comparing learned vec for holdout node to its actual embedding
        pred_vec = best_h[data.test_mask]
        actual_vec = data.y[data.test_mask]
        cosine = cosine_similarity(pred_vec, actual_vec).item()

        # Store results of split
        split_outcome = [data.drug_index[node_id], node_id, cosine, best_loss, best_epoch, time_to_best]
        trace_df.loc[len(trace_df)] = split_outcome
        trace_df.to_csv('LOOCV_cosines.csv', index=False)

        # Store learned vector
        vector_row = [data.drug_index[node_id]] + pred_vec.detach().cpu().numpy().tolist()[0]
        learned_vecs.loc[len(learned_vecs)] = vector_row
        learned_vecs.to_csv(vectors_path, index=False)

# Plot results once done
sns.boxplot(data=trace_df, y='cosine_to_actual')
plt.ylabel('Cosine similarity')
plt.title('LOOCV: Cosine similarity of actual embedding to learned vector')
plt.savefig('LOOCV_cosines.png')
