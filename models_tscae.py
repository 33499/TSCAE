from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from util.crop import del_tensor_ele
from Infonece import InfoNCE
from decoder import TransformerDecoderLayer, TransformerDecoder
class TSCMaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=1, decoder_num_heads=16,
                 dropout=0.5, activation="relu", normalize_before=False, return_intermediate_dec=False,
                 num_queries=224/16*0.75,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_cls_pos = nn.Parameter(torch.zeros(
            1, 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # cross-attention
        decoder_layer = TransformerDecoderLayer(decoder_embed_dim, decoder_num_heads, decoder_embed_dim*mlp_ratio,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_cross = TransformerDecoder(decoder_layer, decoder_depth, decoder_norm,
                                                return_intermediate=return_intermediate_dec)
        # self.decoder_cross_query_embed = nn.Parameter(torch.zeros(
        #     num_queries, 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.norm_pix_loss = norm_pix_loss
        self.MLP_PRE = MLP(embed_dim, patch_size**2 *
                           in_chans, patch_size**2 * in_chans)

        # self.InfoNCE = InfoNCE()
        self.Sim_criterion = nn.CosineSimilarity(dim=1).cuda()
        self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim//4, bias=False),
                                       nn.BatchNorm1d(embed_dim//4),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(embed_dim//4, embed_dim))  # output layer

        # self.predictor_xcls = nn.Sequential(nn.Linear(embed_dim, embed_dim//4, bias=False),
        #                                 nn.BatchNorm1d(embed_dim//4),
        #                                 nn.ReLU(inplace=True), # hidden layer
        #                                 nn.Linear(embed_dim//4, embed_dim)) # output layer
        # --------------------------------------------------------------------------

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.visible_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.visible_token, std=.02)
        torch.nn.init.normal_(self.decoder_cls_pos, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_delete = ids_shuffle[:, len_keep:]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_delete = torch.gather(
            x, dim=1, index=ids_delete.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, x_delete, mask, ids_restore, ids_keep, ids_delete

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        # (1,3,244,244) -> (1,196,1024)
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        N, L, D = x.shape  # batch, length, dim
        # masking: length -> length * mask_ratio
        # (1,49,1024) (1,196) (1,196)
        x_visible, x_delete, mask, ids_restore, ids_keep, ids_delete = self.random_masking(
            x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        # 对x_vidisble添加position
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_delete.shape[1], 1)
        x_v = torch.cat([x_visible, mask_tokens], dim=1)  # no cls token
        x_v = torch.gather(
            x_v, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # 对x_delete添加position
        visible_tokens = self.visible_token.repeat(
            x.shape[0], ids_keep.shape[1], 1)
        x_m = torch.cat([visible_tokens, x_delete], dim=1)  # no cls token
        x_m = torch.gather(
            x_m, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x_v = torch.cat((cls_tokens, x_v), dim=1)
        x_m = torch.cat((cls_tokens, x_m), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x_v = blk(x_v)
        x_v = self.norm(x_v)

        for blk in self.blocks:
            x_m = blk(x_m)
        x_m = self.norm(x_m)
        # delete cls token
        x_v = x_v[:, 1:, :]
        x_m = x_m[:, 1:, :]

        # 用于全局分类任务
        x_m_cls = torch.gather(
            x_m, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_cls = x_m[:, 0, :]

        x_v = torch.gather(
            x_v, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_m = torch.gather(
            x_m, dim=1, index=ids_delete.unsqueeze(-1).repeat(1, 1, D))

        pos_embeds = self.pos_embed[:, 1:, :].repeat(N, 1, 1)

        x_delete_pos_embeds = torch.gather(
            pos_embeds, dim=1, index=ids_delete.unsqueeze(-1).repeat(1, 1, D))
        x_visible_pos_embeds = torch.gather(
            pos_embeds, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_v, x_m, mask, ids_restore, ids_keep, ids_delete, x_delete_pos_embeds, x_visible_pos_embeds, x_m_cls, x_cls

    def forward_crossAttention(self, x_visible, x_delete, x_delete_pos_embeds, x_visible_pos_embeds):
        # 转化weicross_attention需要输入的格式
        query_embed_t = x_delete_pos_embeds.permute(1, 0, 2)  # query的位置编码信息
        decoder_cls_pos = self.decoder_cls_pos.expand(
            -1, query_embed_t.shape[1], -1)
        query_embed = torch.cat((decoder_cls_pos, query_embed_t), 0)
        x_visible = x_visible.permute(1, 0, 2)  # encoder输出的tokens
        x_visible_pos_embed = x_visible_pos_embeds.permute(
            1, 0, 2)  # 输出tokens的位置编码
        tgt = torch.zeros_like(query_embed)  # 用于更新query
        torch.nn.init.normal_(tgt, std=.02)
        mask = None
        hs = self.decoder_cross(tgt, x_visible, memory_key_padding_mask=mask,
                                pos=x_visible_pos_embed, query_pos=query_embed)

        return hs, x_delete

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)  # x: (N, L, patch_size**2 *3)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def sce_loss(self, pre_delete, gt_delete):
        N, L, D = gt_delete.shape  # batch, length, dim
        loss = (pre_delete - gt_delete) ** 2
        loss = loss.mean(dim=-1)
        loss = loss.sum()/(N*L)
        return loss

    def pre_loss(self, pre_delete, imgs, ids_delete):
        pred = self.MLP_PRE(pre_delete)
        target = self.patchify(imgs)  # x: (N, L, patch_size**2 *3)
        N, L, D = target.shape  # batch, length, dim
        target_mask = torch.gather(
            target, dim=1, index=ids_delete.unsqueeze(-1).repeat(1, 1, D))
        loss = (pred - target_mask) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.sum()/(N*L)
        return loss, pred

    def pre_cls_loss(self, x_m_cl, pre_patches_cls):
        b, l, d = x_m_cl.shape
        x_globle = x_m_cl.permute(0, 2, 1).reshape(b, d, 7, 7)

        x_globle = F.max_pool2d(
            x_globle, (x_globle.size(2), x_globle.size(3)))  # 全局特征

        pre_patches_globle = pre_patches_cls.squeeze()
        gt_globle = x_globle.squeeze()

        pre_patches_globle = F.softmax(pre_patches_globle, dim=-1)
        loss = torch.sum(-pre_patches_globle *
                         F.log_softmax(gt_globle, dim=-1), dim=-1).mean()
        return loss

    def pre_clstoken_loss(self, x_cls, pre_patches_cls):

        pre_patches_globle = pre_patches_cls.squeeze()
        gt_globle = x_cls

        pre_patches_globle = F.softmax(pre_patches_globle, dim=-1)
        loss = torch.sum(-pre_patches_globle *
                         F.log_softmax(gt_globle, dim=-1), dim=-1).mean()
        return loss

    def pre_clstoken_loss_position(self, x_cls, pre_patches_cls):

        pre_patches_globle = pre_patches_cls.squeeze()
        gt_globle = x_cls

        gt_globle = F.softmax(gt_globle, dim=-1)
        loss = torch.sum(-gt_globle *
                         F.log_softmax(pre_patches_globle, dim=-1), dim=-1).mean()
        return loss

    def pre_cls_loss_detach(self, x_m_cl, pre_patches_cls):
        b, l, d = x_m_cl.shape
        x_globle = x_m_cl.permute(0, 2, 1).reshape(b, d, 7, 7)

        x_globle = F.max_pool2d(
            x_globle, (x_globle.size(2), x_globle.size(3)))  # 全局特征

        pre_patches_cls = pre_patches_cls.squeeze()
        pre_patches_globle = self.predictor_xcls(pre_patches_cls)
        pre_patches_cls = pre_patches_cls.detach()
        gt_globle = x_globle.squeeze()

        pre_patches_globle = F.softmax(pre_patches_globle, dim=-1)
        loss = torch.sum(-pre_patches_globle *
                         F.log_softmax(gt_globle, dim=-1), dim=-1).mean()
        return loss

    def infoNece_loss(self, pre_delete, gt_delete):
        # batch, length, dim (4,147,1024)
        N, L, D = gt_delete.shape
        query = pre_delete.reshape(N*L, D)
        positive_key = gt_delete.reshape(N*L, D)
        # negative_keys
        b_index = list(torch.arange(N).repeat(N, 1))
        negative_indexs = []
        for i, b in enumerate(b_index):
            negative_indexs.append(del_tensor_ele(b, i))
        negative = [torch.einsum('bld->lbd', gt_delete[negative_index])
                    for negative_index in negative_indexs]
        negative_keys = torch.cat(negative, dim=0)
        loss = self.InfoNCE(query, positive_key, negative_keys)
        return loss

    def simsiam_loss(self, pre_delete, gt_delete):

        # batch, length, dim (4,147,1024)
        N, L, D = gt_delete.shape
        z1 = pre_delete.reshape(N*L, D)
        z2 = gt_delete.reshape(N*L, D)

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC
        z1 = z1.detach()
        z2 = z2.detach()
        loss = -(self.Sim_criterion(p1, z2).mean() +
                 self.Sim_criterion(p2, z1).mean()) * 0.5
        return loss

    def forward_imgs(self, imgs, pred, ids_restore, ids_keep):
        # 解码img
        target = self.patchify(imgs)  # x: (N, L, patch_size**2 *3)
        N, L, D = target.shape
        # 找到可见的patch块
        x_visible = torch.gather(
            target, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # 可见的patch块和不可见的patch块叠加
        x_pre = torch.cat((x_visible, pred), dim=1)

        # 复原到正确的顺序
        x_imges = torch.gather(
            x_pre, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_pre.shape[2]))  # unshuffle

        return x_imges

    def forward(self, imgs, mask_ratio=0.75):
        # (1,50,1024),(1,196)
        x_visible, x_delete, mask, ids_restore, ids_keep, ids_delete, x_delete_pos_embeds, x_visible_pos_embeds, x_m_cls, x_cls = self.forward_encoder(
            imgs, mask_ratio)
        # (4,147,1024),(4,147,1024)
        pre_delete, gt_delete = self.forward_crossAttention(
            x_visible, x_delete, x_delete_pos_embeds, x_visible_pos_embeds)
        # pre_imgs为预测的图像像素(4,147,768)
        pre_patches = pre_delete.permute(1, 0, 2)[:, 1:, :]
        pre_patches_cls = pre_delete.permute(1, 0, 2)[:, :1, :]
        loss_1, pre_imgs = self.pre_loss(pre_patches, imgs, ids_delete)

        loss_2 = self.simsiam_loss(pre_patches, gt_delete)

        loss_3 = self.pre_clstoken_loss_position(x_cls, pre_patches_cls)
        # loss_3 = self.pre_clstoken_loss(x_cls,pre_patches_cls)
        # loss_3 = self.pre_cls_loss_detach(x_m_cls,pre_patches_cls)
        # loss_2 = self.infoNece_loss(pre_delete.permute(1, 0, 2), gt_delete)
        # loss_2 = self.sce_loss(pre_delete.permute(1, 0, 2), gt_delete)

        loss = loss_1 + loss_2 + loss_3

        x_imgs = self.forward_imgs(imgs, pre_imgs, ids_restore, ids_keep)
        return loss, x_imgs, mask


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = TSCMaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = TSCMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = TSCMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512*2, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = TSCMaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512*2, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b  # decoder: 348 dim, 6 blocks
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
