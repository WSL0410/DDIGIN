# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# loaded with transform parameter set as such in order to obtain the adj_t matrix
# required for the GNN layers

import torch

from torch_geometric.data import Data

import torch_geometric.transforms as T

from NV import train_Node2Vec
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import numpy as np

from DDIGIN import GIN
from Predictor import LinkPredictor
from train import train, test

gnn_args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 256,
    'dropout': 0.5,
    'epochs': 100,
    'weight_decay': 1e-5,
    'lr': 0.005,
    'attn_size': 32,
    'num_layers': 3,
    'log_steps': 1,
    'eval_steps': 5,
    'runs': 10,
    'batch_size': 64 * 1024,

}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor()) # loading ogb-ddi
print('Task type: {}'.format(dataset.task_type))
graph = dataset[0]
adj_t = graph.adj_t.to(device)


dataset = PygLinkPropPredDataset(name='ogbl-ddi')
data = dataset[0]

split_edge = dataset.get_edge_split()
train_edges = split_edge['train']['edge']
idx = torch.randperm(split_edge['train']['edge'].size(0))
idx = idx[:split_edge['valid']['edge'].size(0)]
split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}



train_edges_node2vec = train_edges.T
data_node2vec = Data(edge_index=train_edges_node2vec)

filepath = 'training_outputs/train-embedding-256.pt'
node2vec_args = {'device':0, 'embedding_dim':256, 'walk_length':40, 'context_size':20, 'walks_per_node':10,
      'batch_size':256, 'lr':0.01, 'epochs':100, 'log_steps':1}
train_Node2Vec(node2vec_args, data_node2vec, filepath)
filepath = 'training_outputs/train-embedding-256.pt'
pretrained_weight = torch.load(filepath, map_location='cpu').to(device)
pretrained_weight = pretrained_weight.cpu().data.numpy()
node2vec_emb = torch.nn.Embedding(dataset.data.num_nodes, gnn_args['hidden_size']).to(device)
node2vec_emb.weight.data.copy_(torch.from_numpy(pretrained_weight))


def train_model(model, emb, gnn_args, predictor, model_name):
    train_hits_arr, val_hits_arr, test_hits_arr = [], [], []

    evaluator = Evaluator(name='ogbl-ddi')
    for run in range(2):
        max_valhits, train_hits_run, test_hits_run = float('-inf'), 0, 0

        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=gnn_args['lr'])

        for epoch in range(1, 1 + gnn_args['epochs']):
            loss = train(model, predictor, emb.weight, adj_t, split_edge,
                         optimizer, gnn_args['batch_size'])

            if epoch % gnn_args['eval_steps'] == 0:
                results = test(model, predictor, emb.weight, adj_t, split_edge,
                               evaluator, gnn_args['batch_size'])

                if epoch % gnn_args['log_steps'] == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

                # check val-hits@20
                train_hits, valid_hits, test_hits = results['Hits@20']
                if valid_hits >= max_valhits:  # if validhits20 is higher than max, save ckpt
                    max_valhits = valid_hits
                    train_hits_run = train_hits
                    test_hits_run = test_hits
                    # Save model checkpoint for current run.
                    model_path = f"training_outputs/{model_name}.pt"
                    emb_path = f'training_outputs/{model_name}_init_emb.pt'
                    save_model_ckpt(model, emb, optimizer, predictor, loss, emb_path, model_path)
        train_hits_arr.append(train_hits_run)
        test_hits_arr.append(test_hits_run)
        val_hits_arr.append(max_valhits)

    # Print overall stats arrays for best model based on val hits@20
    print("Val_hits@20: ", val_hits_arr)
    print("Test_hits@20: ", test_hits_arr)
    print("Train_hits@20: ", train_hits_arr)

    # Print best model stats (based on val hits@20)
    val_max = max(val_hits_arr)
    print("Best model val hits@20: ", max(val_hits_arr))
    max_idx = val_hits_arr.index(val_max)
    print('Best model test hits@20: ', test_hits_arr[max_idx])
    print('Best model train hits@20: ', val_hits_arr[max_idx])

    # convert to numpy array
    val_hits_arr = np.array(val_hits_arr)
    test_hits_arr = np.array(test_hits_arr)
    train_hits_arr = np.array(train_hits_arr)

    # Print average stats + variance
    print(f"Average best train hits@20: {np.mean(train_hits_arr)}; var: {np.var(train_hits_arr)}")
    print(f"Average best val hits@20: {np.mean(val_hits_arr)}; var: {np.var(val_hits_arr)}")
    print(f"Average best test hits@20: {np.mean(test_hits_arr)}; var: {np.var(test_hits_arr)}")


# %%

def save_model_ckpt(model, emb, optimizer, predictor, loss, emb_path, model_path):
    ''' Save model and embedding checkpoints. '''
    EPOCH = 100
    # Save model params
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'loss': loss,
    }, model_path)
    torch.save(emb.weight.data.cpu(), emb_path)


# %%

def load_model_ckpt(curr_model, model_name, run):
    evaluator = Evaluator(name='ogbl-ddi')
    model_path = f"training_outputs/{model_name}.pt"
    emb_path = f'training_outputs/{model_name}_init_emb.pt'

    pretrained_weight = torch.load(emb_path, map_location='cpu').to(device)
    pretrained_weight = pretrained_weight.cpu().data.numpy()
    emb_after = torch.nn.Embedding(dataset.data.num_nodes, gnn_args['hidden_size']).to(device)
    emb_after.weight.data.copy_(torch.from_numpy(pretrained_weight))

    predictor = LinkPredictor(gnn_args['hidden_size'], gnn_args['hidden_size'], 1,
                              gnn_args['num_layers'], gnn_args['dropout']).to(device)
    optimizer = torch.optim.Adam(
        list(curr_model.parameters()) + list(emb_after.parameters()) +
        list(predictor.parameters()), lr=gnn_args['lr'])

    # Load model, predictor, and optimizer params
    checkpoint = torch.load(model_path)
    curr_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])

    h = curr_model(emb_after.weight, adj_t)
    final_emb_path = f'training_outputs/{model_name}_final_emb_{run}.pt'
    torch.save(h, final_emb_path)

    results = test(curr_model, predictor, emb_after.weight, adj_t, split_edge,
                   evaluator, gnn_args['batch_size'])

    # Print hits stats
    for key, result in results.items():
        print(key)
        train_hits, valid_hits, test_hits = result
        print(f'Train: {100 * train_hits:.2f}%, '
              f'Valid: {100 * valid_hits:.2f}%, '
              f'Test: {100 * test_hits:.2f}%')


# Train GIN with Node2Vec Features
gin_model = GIN(gnn_args['hidden_size'], gnn_args['hidden_size'], gnn_args['hidden_size'],
             gnn_args['num_layers'], gnn_args['dropout']).to(device)

predictor = LinkPredictor(gnn_args['hidden_size'], gnn_args['hidden_size'], 1,
                          gnn_args['num_layers'], gnn_args['dropout']).to(device)
gin_emb_node2vec = torch.nn.Embedding(dataset.data.num_nodes, gnn_args['hidden_size']).to(device)
gin_emb_node2vec.weight.data.copy_(torch.from_numpy(pretrained_weight))
train_model(gin_model, gin_emb_node2vec, gnn_args, predictor, 'gin_node2vec_feat_256')

#%%

# load GIN with Node2Vec Features
gin_model = GIN(gnn_args['hidden_size'], gnn_args['hidden_size'], gnn_args['hidden_size'],
             gnn_args['num_layers'], gnn_args['dropout']).to(device)
load_model_ckpt(gin_model, 'gin_node2vec_feat_256',0)






