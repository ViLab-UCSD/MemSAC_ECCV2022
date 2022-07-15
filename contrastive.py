import torch.nn as nn
import torch.nn.functional as F
import torch

def cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res

class MSCLoss(nn.Module):
    def __init__(self, config_data):
        super().__init__()
        self.ranking_method = 'sim_ratio'
        self.ranking_k = config_data['ranking_k'] #ranking_k = m for sim-ratio(knn confidence measure)
        self.top_ranked_n = config_data['top_ranked_n'] # mu in number
        self.eps = 1e-9
        self.similarity_func = config_data['similarity_func']  # euclidean dist, cosine
        self.top_n_sim = config_data['top_n_sim'] # k for knn
        self.knn_method = config_data['knn_method'] #if 'ranking' use filtering, if 'classic' no filtering
        # self.K = config_data["K"]
        self.batch_size = config_data["batch_size"]
        self.all_assigned = None # used in main file for collecting statistics
        self.conf_ind = None
        self.iter = 0
        self.tau = config_data['tau']
        # self.log_file = "dc_psuedo_and_similarity_ila.txt"
        self.log_file = "real_clipart_msc.txt"
        with open(self.log_file , "w") as fh:
            write_str = "iter\tMeanSimScore\tStdSimScore\tMeanNegScore\tStdNegScore\n"
            fh.write(write_str)

    def __get_sim_matrix(self, out_src, out_tar):
        matrix = None
        if (self.similarity_func == 'euclidean'): ## Inverse Euclidean Distance
            matrix = cdist(out_src, out_tar)
            matrix = matrix + 1.0
            matrix = 1.0/matrix

        elif (self.similarity_func == 'gaussian'): ## exponential Gaussian Distance
            matrix = cdist(out_src, out_tar)
            matrix = torch.exp(-1*matrix)

        elif (self.similarity_func == 'cosine'): ## Cosine Similarity
            out_src = F.normalize(out_src, dim=1, p=2)
            out_tar = F.normalize(out_tar, dim=1, p=2)
            matrix = torch.matmul(out_src, out_tar.T)

        else:
            raise NotImplementedError

        return matrix

    #func to assign target labels by KNN
    def __target_labels_sort_div(self, sim_matrix, src_labels):

        ind = torch.sort(sim_matrix, descending=True, dim=0).indices
        k_orderedNeighbors = src_labels[ind[:self.top_n_sim]]
        assigned_target_labels = torch.mode(k_orderedNeighbors, dim=0).values

        return assigned_target_labels, ind
    
    # #calculate loss
    # def calc_loss_rect_matrix(self, confident_sim_matrix, src_labels, confident_tgt_labels):
    #     n_src = src_labels.shape[0]
    #     n_tgt = confident_tgt_labels.shape[0]
        
    #     vr_src = src_labels.unsqueeze(-1).repeat(1, n_tgt)
    #     hr_tgt = confident_tgt_labels.unsqueeze(-2).repeat(n_src, 1)
        
    #     mask_sim = (vr_src == hr_tgt).float()
    #     sim_sum = torch.sum(mask_sim, dim=1)
    #     valid_source_mask = sim_sum > 0.

    #     expScores = torch.softmax(confident_sim_matrix/self.tau, dim=1)
    #     contrastiveMatrix = ((expScores * mask_sim).sum(1)) / (expScores.sum(1))
    #     contrastiveMatrix = contrastiveMatrix[torch.where(valid_source_mask)] # Remove source samples which have no positives in target

    #     # if self.iter > 52:
    #     #     import pdb
    #     #     pdb.set_trace() 

    #     # pos_encoding = self.pos_encoding[torch.where(valid_source_mask)]
    #     # MSC_loss = -1 * torch.mean(torch.log(contrastiveMatrix) * pos_encoding)
    #     MSC_loss = -1 * torch.mean(torch.log(contrastiveMatrix))
    #     return MSC_loss

    def calc_loss_rect_matrix_targetAnchor(self, confident_sim_matrix, src_labels, confident_tgt_labels):
        n_src = src_labels.shape[0]
        n_tgt = confident_tgt_labels.shape[0]
        
        vr_src = src_labels.unsqueeze(-1).repeat(1, n_tgt)
        hr_tgt = confident_tgt_labels.unsqueeze(-2).repeat(n_src, 1)
        
        mask_sim = (vr_src == hr_tgt).float()

        expScores = torch.softmax(confident_sim_matrix/self.tau, dim=0)
        contrastiveMatrix = (expScores * mask_sim).sum(0) / (expScores.sum(0))
        MSC_loss = -1 * torch.mean(torch.log(contrastiveMatrix + 1e-6))
        
        # if self.iter > 1000:
        #     import pdb; pdb.set_trace()
        # expScores = torch.softmax(confident_sim_matrix/self.tau, dim=0)
        # contrastiveMatrix = (expScores * mask_sim).sum(0) / mask_sim.sum(0)
        # MSC_loss = -1 * torch.mean(torch.log(contrastiveMatrix))


        # if self.iter > 1000:
        #     import pdb; pdb.set_trace()
        # expScores = torch.softmax(confident_sim_matrix/self.tau, dim=0)
        # logexpScores = torch.log(expScores)
        # contrastiveMatrix = (logexpScores * mask_sim).sum(0) 
        # MSC_loss = -1 * torch.mean(contrastiveMatrix)

        if self.iter > 5000:
            ## Now, some stats

            ## Compute the similarities
            posIndex = confident_sim_matrix*mask_sim
            posScores = posIndex[posIndex.nonzero(as_tuple=True)]
            meanSim, stdSim = torch.mean(posScores).item() , torch.std(posScores).item()

            ## Compute Dissimilarities
            negIndex = confident_sim_matrix*(~(mask_sim.bool()))
            negScores = negIndex[negIndex.nonzero(as_tuple=True)]
            meanNeg, stdNeg = torch.mean(negScores).item() , torch.std(negScores).item()

            with open(self.log_file , "a") as fh:
                write_str = "{:05d}\t{:.9f}\t{:.8f}\t{:.9f}\t{:.8f}\n".format(self.iter, meanSim, stdSim, meanNeg, stdNeg)
                fh.write(write_str)
        
        return MSC_loss

    def forward(self, source_features, source_labels, target_features, target_labels=None):

        self.iter += 1
        n_tgt = len(target_features)

        sim_matrix = self.__get_sim_matrix(source_features, target_features)
        # sim_matrix = 1.0/(1.0 + torch.cdist(source_features, target_features)) ## Euclidean Similarity
        flat_src_labels = source_labels.squeeze()

        if target_labels is not None:
            # print(target_labels.unique())
            return self.calc_loss_rect_matrix_targetAnchor(sim_matrix, source_labels, target_labels)

        assigned_tgt_labels, sorted_indices  = self.__target_labels_sort_div(sim_matrix, source_labels)
        self.all_assigned = assigned_tgt_labels

        ranking_score_list = []

        for i in range(0, n_tgt): #nln: nearest like neighbours, nun: nearest unlike neighbours
            nln_mask = (flat_src_labels == assigned_tgt_labels[i]).float()
            sorted_nln_mask = nln_mask[sorted_indices[:,i]].bool()
            nln_sim_r  = sim_matrix[:,i][sorted_indices[:,i][sorted_nln_mask]][:self.ranking_k]

            nun_mask = ~(flat_src_labels == assigned_tgt_labels[i])
            nun_mask = nun_mask.float()
            sorted_nun_mask = nun_mask[sorted_indices[:,i]].bool()
            nun_sim_r  = sim_matrix[:,i][sorted_indices[:,i][sorted_nun_mask]][:self.ranking_k]

            pred_conf_score = (1.0*torch.sum(nln_sim_r)/torch.sum(nun_sim_r)).detach() #sim ratio : confidence score
            ranking_score_list.append(pred_conf_score)

        top_n_tgt_ind = torch.topk(torch.tensor(ranking_score_list), self.top_ranked_n)[1]
        confident_sim_matrix = sim_matrix[:, top_n_tgt_ind]
        confident_tgt_labels = assigned_tgt_labels[top_n_tgt_ind] #filtered tgt labels
        # self.conf_ind = top_n_tgt_ind
        # loss_sourceAnch = self.calc_loss_rect_matrix(confident_sim_matrix, source_labels, confident_tgt_labels)
        loss_targetAnch = self.calc_loss_rect_matrix_targetAnchor(confident_sim_matrix, source_labels, confident_tgt_labels)
        # self.pos_encoding = torch.roll(self.pos_encoding, self.batch_size, 0)
        # return loss_sourceAnch #+ loss_targetAnch
        return loss_targetAnch
        
