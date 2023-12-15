import torch
import torch.nn as nn
import torch.nn.functional as F

class conTea(nn.Module):
    def __init__(self,args,base_encoder):
        super().__init__()

        self.encoder = base_encoder(num_class=args.num_class, feat_dim=128, name=args.arch)

        self.moco_queue=args.queue
        self.low_dim=128
        # create the queue_feature
        self.register_buffer("queue_feat", torch.randn(self.moco_queue, self.low_dim)) # embedding pool
        self.queue_feat = F.normalize(self.queue_feat, dim=0)
        # create the queue_distribution of label
        self.register_buffer("queue_dist", torch.randn(self.moco_queue, args.num_class)) # distribution pool
        self.register_buffer("queue_partY",torch.randn(self.moco_queue, args.num_class)) # partial pool
        self.register_buffer("queue_target",torch.randn(self.moco_queue,1)) # target pool
        # create the queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_feat,keys_dist,keys_partY,keys_target):
        batch_size=keys_feat.shape[0]

        ptr=int(self.queue_ptr)
        assert self.moco_queue % batch_size == 0

        # replace the keys at ptr(dequeue and enqueue)
        self.queue_feat[ptr:ptr+batch_size]=keys_feat
        self.queue_dist[ptr:ptr+batch_size]=keys_dist
        self.queue_partY[ptr:ptr+batch_size]=keys_partY
        self.queue_target[ptr:ptr+batch_size]=keys_target
        ptr=(ptr+batch_size)%self.moco_queue # move pointer

        self.queue_ptr[0]=ptr

    def forward(self,img_w=None ,img_s=None, img_distill=None, partY=None,target=None):
        #
        # compute key_k features
        with torch.no_grad():
            # shuffle keys
            shuffle_ids, reverse_ids = get_shuffle_ids(img_w.shape[0])
            img_w,img_distill,partY=img_w[shuffle_ids],img_distill[shuffle_ids],partY[shuffle_ids]
            # forward through the key encoder
            _, feat_k=self.encoder(img_w)
            output_k, _ = self.encoder(img_distill)
            # compute corrected partial distribution
            output_k=torch.softmax(output_k,dim=1)
            output_k=get_correct_conf(output_k,partY)

            # undo shuffle
            feat_k,output_k,partY=feat_k[reverse_ids],output_k[reverse_ids],partY[reverse_ids] #

        features=torch.cat((feat_k,self.queue_feat.clone().detach()),dim=0)
        partYs=torch.cat((partY,self.queue_partY.clone().detach()),dim=0)
        dists=torch.cat((output_k,self.queue_dist.clone().detach()),dim=0)
        targets=torch.cat((target,self.queue_target.clone().detach()),dim=0)

        # dequeue and enqueue
        self._dequeue_and_enqueue(feat_k,output_k,partY,target)

        return features,partYs,dists,output_k

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def get_correct_conf(un_conf,partY):
    # candidate set
    part_confidence = un_conf * partY
    part_confidence = part_confidence / part_confidence.sum(dim=1).unsqueeze(1).repeat(1, part_confidence.shape[1])

    # non-candidate set
    comp_confidence = un_conf * (1 - partY)
    comp_confidence = comp_confidence / (comp_confidence.sum(dim=1).unsqueeze(1).repeat(1, comp_confidence.shape[1]) + 1e-20)

    comp_max = comp_confidence.max(dim=1)[0].unsqueeze(1).repeat(1, partY.shape[1])
    part_min = ((1 - partY) + part_confidence).min(dim=1)[0].unsqueeze(1).repeat(1, partY.shape[1])

    fenmu = (un_conf * partY).sum(dim=1)
    M = 1.0 / fenmu

    M = M.unsqueeze(1).repeat(1, partY.shape[1])
    a = (M * comp_max) / (M * comp_max + part_min)
    a[a==0]=1
    rec_confidence = part_confidence * a + comp_confidence * (1 - a)

    return rec_confidence


class conStu(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        self.encoder = base_encoder(name=args.arch,head='mlp',num_class=args.num_class)

    def forward(self, img_s,img_distill,eval_only=False):
        _,feat_s = self.encoder(img_s)
        output_s,_=self.encoder(img_distill)

        if eval_only:
            return output_s

        return output_s,feat_s
