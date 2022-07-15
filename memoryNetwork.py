import torch
import torch.nn as nn
import contrastive as cl

class queue_ila(nn.Module):
    def __init__(self, dim, K=65536, m=0.999, T=0.007, knn=5, top_ranked_n=32, ranking_k=4, knn_method="ranking", batch_size=32, similarity_func="euclidean"):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(queue_ila, self).__init__()

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
        # self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.Parameter(data=torch.zeros(K , dim), requires_grad=False)
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_ptr = nn.Parameter(torch.zeros(1,dtype=torch.long), requires_grad=False)

    # @torch.no_grad()
    # def init_queue(self, dset_loader_src):
    #     # Len(Q) <<< Len(Dataset)
    #     iter_source = iter(dset_loader_src)
    #     queue_filled = 0
    #     while(queue_filled < self.K):
    #         try:
    #             inputs_source, labels_source = iter_source.next()
    #         except:
    #             iter_source = iter(dset_loader_src)
    #             inputs_source, labels_source = iter_source.next()
    #         inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
    #         with torch.no_grad():
    #             k, _ = self.key_encoder(inputs_source)
    #             self._dequeue_and_enqueue(k, labels_source)
    #             queue_filled+= k.shape[0]
    #     del inputs_source, labels_source

    # @torch.no_grad()
    # def copy_params(self, encoder_q):
    #     for param_cls_q, param_cls_k in zip(encoder_q.get_parameters(), self.key_encoder.get_parameters()):
    #         for param_q, param_k in zip(param_cls_q["params"], param_cls_k["params"]):
    #             param_k.data.copy_(param_q.data)  # initialize
    #             param_k.requires_grad = False  # not update by gradient

    # @torch.no_grad()
    # def _momentum_update_key_encoder(self, encoder_q):
    #     """
    #     Momentum update of the key encoder
    #     """
        
    #     for param_cls_q, param_cls_k in zip(encoder_q.get_parameters(), self.key_encoder.get_parameters()):
    #         # print(param_cls_k , param_cls_q)
    #         # i=0
    #         for param_q, param_k in zip(param_cls_q["params"], param_cls_k["params"]):
    #             param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    #             # i+=1
    #         # print("Updated %d keys"%(i))
    #     # import pdb
    #     # pdb.set_trace()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, key_labels): #TODO:check key labels
        # gather keys before updating queue

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_labels[ptr:ptr + batch_size] = key_labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    # def forward(self, encoder_q, target_features, source_batch, source_labels):
    def forward(self, features, source_labels, target_labels=None):

        # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     self._momentum_update_key_encoder(encoder_q)  # update the key encoder
        #     self.key_features, _ = self.key_encoder(source_batch)  # keys: NxC
        #     self.key_features = self.key_features.detach()
        source_features = features[:source_labels.shape[0]]
        target_features = features[source_labels.shape[0]:]
        # source_features, target_features = features.split(source_labels.shape[0])
        self.key_features = source_features.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(self.key_features, source_labels)
        sim_loss = self.sim_module(self.queue, self.queue_labels, target_features, target_labels)
        return sim_loss
