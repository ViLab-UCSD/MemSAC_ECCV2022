import torch
import torch.nn as nn
import model.contrastive as cl

class MemoryModule(nn.Module):
    def __init__(self, dim, K=48000, m=0, T=0.007, knn=5, top_ranked_n=32, ranking_k=4, knn_method="ranking", batch_size=32, similarity_func="cosine"):
        """
        dim: feature dimension (default: 128)
        K: queue size (default: 65536)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        sim_config = {
        'similarity_func': similarity_func,
        'top_n_sim'      : knn,
        'ranking_k'      : ranking_k,
        'top_ranked_n'   : top_ranked_n,
        'knn_method'     : knn_method,
        'batch_size'     : batch_size,
        'tau'            : T
            }

        sim_module = cl.MSCLoss(sim_config)
        self.sim_module = sim_module.cuda()

        self.queue_labels = nn.Parameter(data=torch.zeros(K, dtype=torch.long), requires_grad=False)
        self.queue = nn.Parameter(data=torch.zeros(K , dim), requires_grad=False)
        self.queue_ptr = nn.Parameter(torch.zeros(1,dtype=torch.long), requires_grad=False)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, key_labels): 

        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0  ## Just makes life easier. https://github.com/facebookresearch/moco/blob/78b69cafae80bc74cd1a89ac3fb365dc20d157d3/moco/builder.py#L60 

        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_labels[ptr:ptr + batch_size] = key_labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, features, source_labels):

        source_features = features[:source_labels.shape[0]]
        target_features = features[source_labels.shape[0]:]
        self.key_features = source_features.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(self.key_features, source_labels)
        sim_loss = self.sim_module(self.queue, self.queue_labels, target_features)
        return sim_loss
