import torch
import torch.nn.functional as F
import torch.nn as nn

class WeightedConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07,dist_temprature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.dist_temperature = dist_temprature

    def forward(self, features, dist, partY, args,epoch,mask=None, batch_size=-1):
        if mask is not None:
            mask=mask.float()
            #compute logits
            anchor_dot_contrast=torch.div(
                torch.matmul(features[:batch_size],features.T),
                self.temperature
            )
            # for numerical stability
            logits_max,_=torch.max(anchor_dot_contrast,dim=1,keepdim=True)
            logits=anchor_dot_contrast-logits_max.detach()

            # mask-out self-contrast cases
            logits_mask=torch.scatter(
                torch.ones_like(anchor_dot_contrast),
                1,
                torch.arange(batch_size).view(-1,1).cuda(),
                0
            )
            mask=logits_mask*mask

            # compute weight
            dist_temperature=self.dist_temperature

            dist_norm = dist / torch.norm(dist, dim=-1, keepdim=True)
            anchor_dot_simi = torch.div(torch.matmul(dist_norm[:batch_size], dist_norm.T), dist_temperature)
            # for numerical stability
            logits_simi_max, _ = torch.max(anchor_dot_simi, dim=1, keepdim=True)
            logits_simi = anchor_dot_simi - logits_simi_max.detach()
            exp_simi=torch.exp(logits_simi)*mask
            weight=exp_simi/exp_simi.sum(dim=1).unsqueeze(1).repeat(1, anchor_dot_simi.shape[1])

            # compute log_prob
            exp_logits=torch.exp(logits)*logits_mask
            log_prob=logits-torch.log(exp_logits.sum(1,keepdim=True)+1e-12)

            # compute weighted of log-likelihood over positive
            weighted_log_prob_pos=weight*log_prob

            #loss
            loss=-(self.temperature/self.base_temperature)*weighted_log_prob_pos
            loss=loss.sum(dim=1).mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            k, queue = k.detach(), queue.detach()
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss


def CE_loss(output, confidence,Y=None):
    logsm_outputs = F.log_softmax(output, dim=1)
    final_outputs = logsm_outputs * confidence
    average_loss = - ((final_outputs).sum(dim=1)).mean()
    return average_loss