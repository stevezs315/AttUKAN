import torch
import torch.nn.functional as F
import torch.nn as nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, threshold=0.1, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, contrastive_method='simclr'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        # self._cosine_similarity = torch.nn.CosineSimilarity(dim=2) #dim=2？
        # 原代码：
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        self.threshold = threshold
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)

        # print("x shape: {}".format(x.shape))
        # print("y shape: {}".format(y.shape))

        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))

        # print("v shape: {}".format(v.shape))

        return v

    def forward(self, features, labels=None, mask=None, weight=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
        输出:
            loss值
        """
        device = features.device

        original_mask = mask

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

            # print("after view features shape: {}".format(features.shape))
            # print("after view features: {}".format(features))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            if self.contrastive_method in ['gcl']:
                mask = torch.eq(labels, labels.T).float().to(device)
            elif self.contrastive_method in ['pcl', 'wpcl', 'wcl', 'superpixel_pcl']:
                mask = (torch.abs(
                    labels.T.repeat(batch_size, 1) - labels.repeat(1, batch_size)) < self.threshold).float().to(device)

                # breakpoint()
        else:  # wcl, wpcl
            mask = mask.float().to(device)

        # print("simclr mask: {}".format(mask))
        # raise ValueError
        # print("labels: {}".format(labels))
        # print("mask: {}".format(torch.abs(labels.T.repeat(batch_size, 1) - labels.repeat(1, batch_size))))
        # raise ValueError
        # print("mask \n:{}".format(mask))
        # breakpoint()
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''

        contrast_count = features.shape[1]  #
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # torch.unbind(dim=1)按列拆开,也就是一个batch的放在一起

        # print("contrast_count :{}".format(contrast_count))
        # print("contrast_feature :{}".format(contrast_feature.shape))
        # print("torch.unbind: {}".format(temp))
        # print("contrast feature shape: {}".format(contrast_feature.shape))
        # breakpoint()

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            # print("anchor_feature shape: {}".format(anchor_feature.shape))
            # print("contrast_feature shape: {}".format(contrast_feature.shape))
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        logits = torch.div(
            self._cosine_simililarity(anchor_feature, contrast_feature),
            self.temperature)  # 前者除以后者

        # print("logits: {}".format(logits))

        # SupConLoss原代码内容：
        # compute logits
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)
        # print("anchor_dot_contrast: {}".format(anchor_dot_contrast))
        # # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits_2 = anchor_dot_contrast - logits_max.detach()

        # print("logits_2: {}".format(logits_2))

        # tile mask
        # print("mask1 shape: {}".format(mask.shape))
        if labels == None and original_mask != None:
            mask = mask
        else:
            mask = mask.repeat(anchor_count, contrast_count)

        # print("mask_repeat shape: {}".format(mask.shape))
        # raise ValueError
        # print("mask_repeat : {}".format(mask))

        # print("mask shape: {}".format(mask.shape))
        # breakpoint()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # print("positive mask: {}".format(mask))

        # breakpoint()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive

        # print("mask: {}".format(mask))
        # print("mask shape: {}".format(mask.shape))

        # print("weight: {}".format(weight))
        # for i in range(0, batch_size * 2):
        #     print("mask * weight: {}".format(((mask * weight)/ (mask * weight).sum(1))[:,i]))
        #     print("\n")
        # raise ValueError

        if weight == None:
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        else:
            mean_log_prob_pos = (mask * weight * log_prob).sum(1) / (mask * weight).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def ConLoss(feature, label_original):
    bsz = feature.shape[0]  # batchsize
    f1 = feature # feature [Batchsize, Channel, Height, Width]

    label_original = label_original
    fmap_h = feature.shape[2]
    fmap_w = feature.shape[3]

    label = F.interpolate(label_original.float().cuda(), size=(fmap_h, fmap_w), mode='bilinear',align_corners=True)

    label = label.squeeze(1)  # 降采样之后的标签

    criterion = SupConLoss(temperature=0.1, contrastive_method='gcl').cuda()
    loss_sp_intra = 0
    #     #-------------------方法一-----------------------#
    f1 = torch.transpose(f1.view(f1.shape[1], -1), 0, 1)
    sp_features = torch.cat([f1.unsqueeze(1), f1.unsqueeze(1)], dim=1)
    #print("f1: {}".format(f1.shape))
    sp_label1 = label.view(-1)
    #print("label_sp_1: {}".format(sp_label1.shape))

    size = f1.shape[0] //10

    for i in range(10):
        single_sp_features = sp_features[i * size : (i+1) * size]
        # print(single_sp_features.shape)
        single_sp_label1 = sp_label1[i * size : (i+1) * size]
        # print(single_sp_label1.shape)
        loss_sp_intra += criterion(single_sp_features, labels=single_sp_label1)
    loss_sp_intra = loss_sp_intra / 10
    # loss_sp_intra += criterion(sp_features, labels=sp_label1)
    return loss_sp_intra / bsz  # 这个loss就是和BCE loss放在一起做梯度反传的