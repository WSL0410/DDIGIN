
import torch
from torch_geometric.nn import Node2Vec


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

# save
def save_embedding(model, filepath):
    torch.save(model.embedding.weight.data.cpu(), filepath)


def train_Node2Vec(args, data, filepath):
    model = Node2Vec(data.edge_index, args['embedding_dim'], args['walk_length'],
                    args['context_size'], args['walks_per_node'],
                    sparse=True).to(device)

    loader = model.loader(batch_size=args['batch_size'], shuffle=True,
                        num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args['lr'])

    model.train()
    for epoch in range(1, args['epochs'] + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args['log_steps'] == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 100 == 0:  # Save model every 100 steps.
                save_embedding(model, filepath)
        save_embedding(model, filepath)
