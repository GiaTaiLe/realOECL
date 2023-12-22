import diffdist.functional as distops
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    """
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    """

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix


def get_l2_matrix(outputs, chunk=2, multi_gpu=False):
    """
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    """

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    l2_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return l2_matrix


def get_norm(outputs, multi_gpu=False):
    """
    :param outputs:
    :param multi_gpu:
    :return: norm of outputs
    """
    if multi_gpu:
        gather_t = [torch.empty_like(outputs) for _ in range(dist.get_world_size())]
        outputs = torch.cat(distops.all_gather(gather_t, outputs))

    norm = torch.norm(outputs, p=2, dim=1)

    return norm


def nt_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    """
        Compute nt_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    """

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    # log-sum trick for numerical stability
    # logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    # sim_matrix = sim_matrix - logits_max.detach()

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss


class NTXent(nn.Module):
    """Wrap a module to get self.training member."""

    def __init__(self, multi_gpu=True, chunk=2):
        super(NTXent, self).__init__()
        self.multi_gpu = multi_gpu
        self.chunk = chunk

    def forward(self, embedding, temperature=0.5):
        """NT-XENT Loss from SimCLR

        :param embedding: embedding of [img, aug_img]
        :param temperature: nce normalization temp
        :returns: scalar loss
        :rtype: float32

        """
        # normalize both embeddings
        embedding = F.normalize(embedding, dim=-1)
        similarity_matrix = get_similarity_matrix(outputs=embedding, multi_gpu=self.multi_gpu)
        loss_sim = nt_xent(similarity_matrix, temperature=temperature)

        return torch.mean(loss_sim)


class NTXentOE(nn.Module):
    def __init__(self, multi_gpu=True, chunk=2):
        super(NTXentOE, self).__init__()
        self.multi_gpu = multi_gpu
        self.chunk = chunk

    def forward(self, embedding, embedding_oe, temperature=0.5, alpha=1.):
        """NT-XENT Loss ver 2 from SimCLR
        :param embedding_oe:
        :param embedding: embedding of [img, aug_img]
        :param temperature: nce normalization temp
        :returns: scalar loss
        :rtype: float32

        """
        # normalize embeddings
        id_norm = get_norm(outputs=embedding)
        oe_norm = get_norm(outputs=embedding_oe)
        #loss_norm = torch.mean(oe_norm) + alpha*torch.mean(-torch.log(1+1e-9-torch.exp(-id_norm)))
        loss_norm = alpha*torch.mean(oe_norm)
        embedding = F.normalize(embedding, dim=-1)
        similarity_matrix = get_similarity_matrix(outputs=embedding, multi_gpu=self.multi_gpu)
        # embedding_oe remains (be not normalized)
        loss_sim = nt_xent(similarity_matrix, temperature=temperature)

        return torch.mean(loss_sim) + loss_norm, torch.mean(id_norm), torch.mean(oe_norm)
