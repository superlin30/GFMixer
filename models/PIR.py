import torch
import numpy as np
import torch.nn as nn
from . import GFMixer,G4P,PatchTST
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import copy
import time


class QualityEstimator(nn.Module):
    def __init__(self, configs):
        super(QualityEstimator, self).__init__()
        self.seq_proj = nn.Linear(configs.seq_len, configs.refine_d_model)
        self.pred_proj = nn.Linear(configs.pred_len, configs.refine_d_model)
        self.activation = nn.Sigmoid()
        self.loss_estimation = nn.Sequential(
            nn.Linear(3 * configs.refine_d_model, configs.refine_d_model),
            nn.GELU(),
            nn.Linear(configs.refine_d_model, 1),
            nn.ReLU()
        )
        self.quality_estimation = nn.Sequential(
            nn.Linear(1, 1),
            self.activation
        )
        self.retrieval_weight = nn.Sequential(
            nn.Linear(configs.retrieval_num + 1, configs.refine_d_model),
            nn.GELU(),
            nn.Linear(configs.refine_d_model, 1),
            self.activation
        )

        self.quality_estimation[0].weight = nn.Parameter(torch.ones(1, 1))
        self.quality_estimation[0].bias = nn.Parameter(torch.zeros(1, ))
        # 特定初始化 (Hard-coded Initialization)：这是一个非常明显的人工干预。nn.Linear(1, 1) 的公式是 $y = w \cdot x + b$。初始化 $w=1, b=0$，意味着在训练刚开始时，线性层什么都不做，直接把输入传过去 ($y = x$)。

    def forward(self, x_enc, x_pred, sims, channel_indicator):

        # x_enc，输入，7，720
        # x_pred，骨干模型的预测，7，720

        x_enc = self.seq_proj(x_enc.permute(0, 2, 1))
        # input，seq_proj = nn.Linear，512维度
        print('*******QualityEstimator里面得到的x_enc.shape*********:',x_enc.shape)

        pred_enc = self.pred_proj(x_pred.permute(0, 2, 1))
        print('*******QualityEstimator里面得到的pred_enc.shape*********:',pred_enc.shape)
        # 模型初步预测的未来序列，pred_proj = nn.Linear
        #每个变量（Channel）独立进行
        # nn.Linear，压缩感知，形状特征

        #这两步线性投影，把“真实输入”和“初步预测”拉到了同一个特征空间里。

        loss_estimated = self.loss_estimation(torch.cat([x_enc, pred_enc, channel_indicator], dim=-1))
        print('*******QualityEstimator里面得到的loss_estimated.shape*********:',loss_estimated.shape)
        # 通道独立评估的误差估计

        alpha = self.quality_estimation(loss_estimated).permute(0, 2, 1)
        # 预估误差(线性学习调整)
        print('*******QualityEstimator里面得到的alpha.shape*********:',alpha.shape)
        
        beta = self.retrieval_weight(torch.cat([loss_estimated, sims], dim=-1)).permute(0, 2, 1)
        # 预估误差+Sims 的线性融合决策。
        print('*******QualityEstimator里面得到的beta.shape*********:',beta.shape)

        return loss_estimated, alpha, beta


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_c = configs.enc_in
        self.including_time_features = configs.including_time_features
        self.retrieval_stride = configs.retrieval_stride
        self.use_norm = configs.use_norm

        self._build_model()
        self.refine_embedding = nn.Linear(configs.pred_len, configs.refine_d_model)
        # configs.refine_d_model是一个可调参数，一般为512

        self.refiner = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.refine_d_model,
                        configs.n_heads),
                    configs.refine_d_model,
                    configs.refine_d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.refine_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.refine_d_model)
        )
        self.refine_projection = nn.Linear(configs.refine_d_model, configs.pred_len)
        self.quality_estimator = QualityEstimator(configs)
        self.channel_indicator = nn.Parameter(
            torch.randn(configs.enc_in, configs.refine_d_model) + torch.ones(configs.enc_in, configs.refine_d_model))
        self.retrieval_mode = 'series'
        self.retrieval_num = configs.retrieval_num



        #self.time_projection_point = nn.Linear(5, 1)
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        t_dim = freq_map.get(configs.freq, 5) # 默认取 5，或者通过 configs 传入具体数值
        self.time_projection_point = nn.Linear(t_dim, 1)
        # 自适应缩放



        self.time_projection_temporal = nn.Linear(configs.pred_len, configs.refine_d_model)
        self.time_backbone, self.time_retrieval, self.time_revision = 0, 0, 0

    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'GFMixer': GFMixer,      # ← 注册新骨干
            'G4P': G4P,      # ← 注册新骨干
        }
        config_model = copy.deepcopy(self.configs)
        self.model = model_dict[self.configs.backbone].Model(config_model).float()

    def construct_index(self, num):  # 建立空的
        key_len = self.seq_len if self.retrieval_mode == 'series' else self.configs.d_model
        #情况 A ('series')：key_len = self.seq_len。
        #解释：直接存储原始时间序列数据作为 Key。检索时，是比较“输入序列”和“历史序列”的相似度。
        #情况 B (其他，如 'latent')：key_len = self.configs.d_model。
        #解释：存储模型 Encoder 输出的高维特征向量作为 Key。检索时，是比较“语义特征”的相似度。

        self.keys = torch.zeros(num, key_len, self.in_c, device=self.channel_indicator.device)
        #Key 是用来做匹配的（比如：过去96个时刻长这样）。

        self.values = torch.zeros(num, self.pred_len, self.in_c, device=self.channel_indicator.device)
        #Value 是检索的目标（比如：这种情况下，未来96个时刻实际上变成了那样）。
        self.index = 0

    @torch.no_grad()
    def add_key_value(self, x_enc, y, index):
        # 把key/value 写到 self.keys[index]、self.values[index]
        bs = x_enc.shape[0]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            y = (y - means) / stdev
            # 【关键点】 Value (y) 也要归一化，而且是用 Key (x) 的统计量！
            # 为什么？因为我们希望存储在库里的 Value 是一个“相对变化量”。
            # 当我们以后检索时，如果 Query 的均值是 100，我们把这个相对 Value 取出来，
            # 再加上 100，就能还原出适合当前 Query 的预测值。


        if self.retrieval_mode == 'series':
            # 实际路径
            x_key = x_enc
        elif self.retrieval_mode == 'embedding':
            x_key = self.model.encode(x_enc)
        else:
            raise NotImplementedError
        self.keys[index, :, :] = x_key
        self.values[index, :, :] = y
        self.index += bs
        torch.cuda.empty_cache()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, index=None, mode='pretrain', timing=False):
        bs = x_dec.shape[0]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if mode == 'pretrain':
            dec_out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.use_norm:
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            return dec_out

        else:
            # 路径从这儿开始：
            t0 = time.time()
            intermediate_results = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 1.骨干模型预测结果，来自ori_model

            t1 = time.time()
            retrieval_results, sims, t = self.retrieval(x_enc, index)
            # 2.进入retrieval

            print('*******forward里面得到的retrieval_results.shape*********:',retrieval_results.shape)

            t2 = time.time()

            # retrieval_results，是单对单的相似度匹配结果，sims是基于k的相似度

            refine_enc = self.refine_embedding(intermediate_results.permute(0, 2, 1))
            print('*******forward里面得到的初始refine_enc.shape*********:',refine_enc.shape)
            # refine_embedding线性层


            if self.including_time_features:
                # 路径

                time_embedding_point = self.time_projection_point(x_mark_dec).permute(0, 2, 1)

                # x_mark_dec示例
                # 时间步 (Time Step)	   Month (月)	Day (日)	Weekday (星期)	Hour (时)
                #         0 (预测的第1小时)	1	       20	      5 (周六)	     12
                #         1 (预测的第2小时)	1	       20	      5 (周六)	     13
                #         2 (预测的第3小时)	1	       20	      5 (周六)	     14

                #（假设有5个时间特征，如月、日、时、分等）。
                print('*******forward里面得到的x_mark_dec前三步示例*********:',x_mark_dec[0, :3, :])

                print('*******forward里面得到的time_embedding_point.shape*********:',time_embedding_point.shape)

                time_embedding = self.time_projection_temporal(time_embedding_point)
                # 将时间维度的 Pred_Len 压成 D_model -> [Batch, 1, D_model]
                print('*******forward里面得到的time_embedding.shape*********:',time_embedding.shape)
                # 压缩成一个非可逆“时间令牌 (Time Token)”。这个 Token 代表了唯一的时间标识：“未来的这段时间是1月20号的中午”。

                refine_enc = torch.cat([time_embedding, refine_enc], dim=1)
                # (Position encoding)
                print('*******forward里面得到的加入时间信息的refine_enc.shape*********:',refine_enc.shape)

                refine_out, _ = self.refiner(refine_enc)
                #self.refiner 是一个 Transformer Encoder（注意力机制）。
                #相互交流 (Self-Attention)：所有的 Token 开始互相“看”。
                #变量看时间：“油温专家”看了一眼“报时员”，发现现在是夏天中午，于是心里想“那我预测的值应该调高一点”。 外生变量
                #变量看变量：“油温专家”看了一眼“负荷专家”，发现负荷很高，于是想“那我的温度也该随之升高”。   协变量



                refine_out = self.refine_projection(refine_out)[:, 1:, :].permute(0, 2, 1)
                print('*******forward里面得到refine_out.shape*********:',refine_out.shape)
                # 线性层变换回去



            else:
                refine_out, _ = self.refiner(refine_enc)
                refine_out = self.refine_projection(refine_out).permute(0, 2, 1)

            loss_estimated, alpha, beta = self.quality_estimator(x_enc, intermediate_results, sims,
                                                                 self.channel_indicator.unsqueeze(0).repeat(bs, 1, 1))
            # 重点



            dec_out = intermediate_results + alpha * refine_out + beta * retrieval_results
            # (Alpha)：
            # 如果 loss_estimated 高 $\rightarrow$ $\alpha$ 变大 $\rightarrow$ Refiner (神经网络修正) 介入。
            # 潜台词：“这题我主干网络不会做，请专家组（Refiner）来帮帮忙。（Local）

            #  (Beta)：
            # 如果 loss_estimated 高 且 sims (检索相似度) 也高 (历史检索) 介入。潜台词：“这题我不会做，但我看历史上第 500 号卷子跟这个很像，快把它的答案抄过来。” （GLoBal)

            t3 = time.time()
            if timing:
                self.time_backbone += t1 - t0
                self.time_retrieval += t2 - t1
                self.time_revision += t3 - t2
            if self.use_norm:
                dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

                intermediate_results = intermediate_results * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                intermediate_results = intermediate_results + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

            return dec_out, (intermediate_results, loss_estimated)

    def retrieval(self, x, index):
        bs = x.shape[0]
        k = self.retrieval_num
        if self.retrieval_mode == 'series':
            # 真实路径
            queries = x
        else:
            queries = self.model.encode(x)
        keys = self.keys
        # keys = self.keys.transpose(2, 1).reshape(-1, self.seq_len)
        t0 = time.time()
        dis = self.cosine_similarity(queries, keys)
        print('*******dis.shape*********:',dis.shape)
        # 计算余弦相似度 (Cosine Similarity)
        # 这是一个巨大的矩阵运算
        # queries: [Batch, Seq_Len, Channel]
        # keys:    [Total_Samples, Seq_Len, Channel]
        # 返回 dis: [Channel, Batch, Total_Samples] -> 每个变量独立计算相似度

        #dis 存的是 未排序的、原始的相似度分数 (Raw Similarity Scores)

        if self.training:
            # offline
            # 算出当前样本自己在数据库里的位置附近的所有索引
            self_range = torch.arange(-self.configs.seq_len, self.configs.seq_len + 1, device=x.device).unsqueeze(0)
            invalid_index = index.unsqueeze(1) + self_range
            invalid_index = invalid_index // self.retrieval_stride
            invalid_index[torch.where(invalid_index < 0)] = 0
            invalid_index[torch.where(invalid_index >= self.index)] = self.index - 1
            row_idx = torch.arange(x.shape[0]).unsqueeze(1).repeat(1, 2 * self.configs.seq_len + 1)
            dis[:, row_idx, invalid_index] = -100
            # 将这些位置的相似度设为负无穷，防止作弊
            # 在训练集上跑检索时，当前的 x 本身就在 self.keys 数据库里！如果不处理，模型肯定会检索到它自己（相似度为1），或者它紧挨着的前后时刻样本。这叫 Data Leakage（数据穿越）。


        dis_topk, indices_topk = torch.topk(dis, dim=2, k=k)
        print('*******dis_topk.shape*********:',dis_topk.shape)
        # dis_topk为最大的k个相似度分数):
        print('*******indices_topk.shape*********:',indices_topk.shape)
        # indices_topk为最大的k个相似度分数的数据库索引):


        sims = dis_topk.permute(1, 0, 2) # bs*c*k
        probs_topk = torch.softmax(dis_topk, dim=2).unsqueeze(-1)  # c*bs*k*1
        #把相似度分数转换成概率分布，和为 1。
        #例子: 假设 Top-3 分数是 [10, 10, 5] (未归一化)，softmax 后变成 [0.49, 0.49, 0.02]。这意味着前两个样本很重要，第三个样本虽然进了 Top-3 但不太重要。
        t = time.time()-t0


        values = self.value_permute  # [in_c, N, pred_len]
    

        # reshape 为 [1, in_c, N, pred_len]，为 batch gather 做准备
        values = values.unsqueeze(0)  # [1, in_c, N, pred_len]

        # indices_topk.shape = [bs, in_c, k]
        # 需要扩展为 [bs, in_c, k, 1] 以便 gather
        indices = indices_topk.permute(1, 0, 2).unsqueeze(-1)  # [in_c, bs, k, 1]

        # 转换 values 为 [in_c, 1, N, pred_len] 以与 indices 对齐
        values = values.expand(bs, -1, -1, -1)  # [in_c, 1, N, pred_len]

        # gather
        values = torch.gather(values, 2, indices.expand(-1, -1, -1, values.size(-1))).permute(1,0,2,3)  # [in_c, bs, k, pred_len]

        output = torch.sum(probs_topk * values, dim=2).permute(1, 2, 0)  # weighted-sum ver
        print('*******retrieval里面最终输出的output.shape*********:',output.shape)
        # 基于检索生成的output的K个样本的加权平均，因此最终只有一个样本，单对单

        print('*******retrieval里面最终输出的sims.shape*********:',sims.shape)
        # 基于检索生成的output的K个样本的相似度，k=10
        return output, sims, 0

    def cosine_similarity(self, queries, keys):
        # equals to person_similarity when revin=True, since std=1, mean=0
        if len(queries.shape) == 2:  # B*L
            q_norm = torch.nn.functional.normalize(queries, p=2, dim=-1)
            k_norm = torch.nn.functional.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.t())
        elif len(queries.shape) == 3:  # B*L*C
            queries = queries.permute(2, 0, 1)
            keys = keys.permute(2, 0, 1)
            q_norm = torch.nn.functional.normalize(queries, p=2, dim=-1)
            k_norm = torch.nn.functional.normalize(keys, p=2, dim=-1)
            return torch.matmul(q_norm, k_norm.permute(0, 2, 1))
