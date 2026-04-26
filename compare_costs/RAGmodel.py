import torch
import torch.nn as nn
from global_configs import *
from utils.utils import *
from src.subnet import *
import torch.nn.functional as F
import math
from torch.nn import functional as F, init



class RAGModel(nn.Module):
    def __init__(self, args):
        super(RAGModel, self).__init__()
            
        self.v_convert_layer = nn.Linear(args.v_dim, args.prompt_dim)
        self.a_convert_layer = nn.Linear(args.a_dim, args.prompt_dim)
        self.t_convert_layer = nn.Linear(args.t_dim, args.prompt_dim)

        self.l2a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.l2v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.l2l = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.v2a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.v2l = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.v2v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.a2v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.a2l = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.a2a = MLPLayer(args.prompt_dim, args.prompt_dim)

        self.cat02t = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat12t = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat22t = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat02a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat12a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat22a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat02v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat12v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat22v = MLPLayer(args.prompt_dim, args.prompt_dim)

        self.a_lvp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.v_alp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.l_avp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )


        self.t_tavp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.a_tavp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.v_tavp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )


        # self.modality_prompt = nn.Parameter(torch.randn((3,args.prompt_dim,args.prompt_len)))
        self.modality_prompt = nn.Parameter(
            torch.empty((3,args.prompt_dim,args.prompt_len))
        )
        for prompt in self.modality_prompt:
            init.kaiming_uniform_(prompt, a=math.sqrt(5))
        
        self.sample_prompt = nn.Parameter(
            torch.empty((3,args.prompt_dim,args.prompt_len))
        )
        for prompt in self.sample_prompt:
            init.kaiming_uniform_(prompt, a=math.sqrt(5))
        self.t_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.t_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.a_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.a_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.v_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.v_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)

        t_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim*2,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.t_encoder = nn.TransformerEncoder(
            encoder_layer=t_encoder_layer,
            num_layers=args.layers
        )
        a_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim*2,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=a_encoder_layer,
            num_layers=args.layers
        )
        v_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim*2,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=v_encoder_layer,
            num_layers=args.layers
        )

        self.classifier = T5ClassificationHead(args)

        # self.t_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.a_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.v_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)

        # self.t_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.a_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.v_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)

        # self.t_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.a_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.v_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)


    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def get_modality_data(self, x_l, x_a, x_v):
        x_l, x_a, x_v = x_l.unsqueeze(dim=0), x_a.unsqueeze(dim=0), x_v.unsqueeze(dim=0)

        #get text modality prompt
        x_l_p = torch.cat(
            [self.modality_prompt[0, :, :], self.l2l(x_l)[0] ,self.a2l(x_a)[0], self.v2l(x_v)[0]],
            # [self.modality_prompt[0, :, :], x_l[0] ,x_a[0], x_v[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_l_p = self.l_avp(x_l_p.transpose(1, 2)).transpose(1, 2) 
        x_l_p = self.l_avp(x_l_p).transpose(1, 2) 

        #get audio modality prompt
        x_a_p = torch.cat(
            [self.modality_prompt[1, :, :], self.a2a(x_a)[0],self.l2a(x_l)[0], self.v2a(x_v)[0]],
            # [self.modality_prompt[1, :, :], x_a[0],x_l[0], x_v[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_a_p = self.a_lvp(x_a_p.transpose(1, 2)).transpose(1, 2) 
        x_a_p = self.a_lvp(x_a_p).transpose(1, 2) 

        #get visual modality prompt
        x_v_p = torch.cat(
            [self.modality_prompt[2, :, :], self.v2v(x_v)[0], self.l2v(x_l)[0], self.a2v(x_a)[0]],
            # [self.modality_prompt[2, :, :], x_v[0], x_l[0], x_a[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_v_p = self.v_alp(x_v_p.transpose(1, 2)).transpose(1, 2) 
        x_v_p = self.v_alp(x_v_p).transpose(1, 2) 

        return x_l_p, x_a_p, x_v_p

    def pad_tensor(self,tensor, target_dim):
        padding_size = target_dim - tensor.size(1)
        if padding_size > 0:
            padding = torch.zeros((tensor.size(0), padding_size), dtype=tensor.dtype).to(DEVICE)
            tensor = torch.cat((tensor, padding), dim=1)
        return tensor

    def cal_cos_similarity(self,x_t,x_a,x_v,x_t_re,x_a_re,x_v_re,label = None):
        """
        :param x_t: Shape (batch_size, sequence_length, feature_dim)
        :param x_a: Shape (batch_size, sequence_length, feature_dim)
        :param x_v: Shape (batch_size, sequence_length, feature_dim)
        :return: tuple of tensors containing the most similar samples for each modality
        """
        if len(x_t_re.shape) < 3:
            x_t_re = x_t_re.unsqueeze(0)
            x_a_re = x_a_re.unsqueeze(0)
            x_v_re = x_v_re.unsqueeze(0)
        
    

        x_t_flat = x_t.reshape(x_t.size(0), x_t.size(1)*x_t.size(2))  # (batch_size, seq_len * feature_dim)
        x_a_flat = x_a.reshape(x_a.size(0), x_a.size(1)*x_a.size(2))  # (batch_size, seq_len * feature_dim)
        x_v_flat = x_v.reshape(x_v.size(0), x_v.size(1)*x_v.size(2))  # (batch_size, seq_len * feature_dim)

        max_dim = max(x_t_flat.shape[1],x_a_flat.shape[1],x_v_flat.shape[1])
        x_t_flat = self.pad_tensor(x_t_flat,max_dim)
        x_a_flat = self.pad_tensor(x_a_flat,max_dim)
        x_v_flat = self.pad_tensor(x_v_flat,max_dim)

        x_t_re_flat = x_t_re.reshape(x_t_re.size(0), x_t_re.size(1)*x_t_re.size(2))  # (batch_size, seq_len * feature_dim)
        x_a_re_flat = x_a_re.reshape(x_a_re.size(0), x_a_re.size(1)*x_a_re.size(2))  # (batch_size, seq_len * feature_dim)
        x_v_re_flat = x_v_re.reshape(x_v_re.size(0), x_v_re.size(1)*x_v_re.size(2))  # (batch_size, seq_len * feature_dim)

        max_dim = max(x_t_re_flat.shape[1],x_a_re_flat.shape[1],x_v_re_flat.shape[1])
        x_t_re_flat = self.pad_tensor(x_t_re_flat,max_dim)
        x_a_re_flat = self.pad_tensor(x_a_re_flat,max_dim)
        x_v_re_flat = self.pad_tensor(x_v_re_flat,max_dim)
        
        # 2. 计算每个模态的余弦相似度
        cosine_sim_1 = F.cosine_similarity(x_t_flat.unsqueeze(1), x_t_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        cosine_sim_2 = F.cosine_similarity(x_a_flat.unsqueeze(1), x_a_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        cosine_sim_3 = F.cosine_similarity(x_v_flat.unsqueeze(1), x_v_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        cosine_sim_4 = F.cosine_similarity(x_t_flat.unsqueeze(1), x_a_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        cosine_sim_5 = F.cosine_similarity(x_t_flat.unsqueeze(1), x_v_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        cosine_sim_6 = F.cosine_similarity(x_a_flat.unsqueeze(1), x_t_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        cosine_sim_7 = F.cosine_similarity(x_a_flat.unsqueeze(1), x_v_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        cosine_sim_8 = F.cosine_similarity(x_v_flat.unsqueeze(1), x_t_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        cosine_sim_9 = F.cosine_similarity(x_v_flat.unsqueeze(1), x_a_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        
        if label is None :
            if x_t_re.shape[0] == 1:
                _, most_similar_1 = torch.topk(cosine_sim_1, k=1, dim=1,largest=True)
                most_similar_1_temp = most_similar_1[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_4 = torch.topk(cosine_sim_4, k=1, dim=1,largest=True)
                most_similar_4 = most_similar_4[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_5 = torch.topk(cosine_sim_5, k=1, dim=1,largest=True)
                most_similar_5 = most_similar_5[:, 0]  # 获取除自己外最相似的索引
            else:
                _, most_similar_1 = torch.topk(cosine_sim_1, k=2, dim=1,largest=True)
                most_similar_1_temp = most_similar_1[:, 1]  # 获取除自己外最相似的索引
                _, most_similar_4 = torch.topk(cosine_sim_4, k=2, dim=1,largest=True)
                most_similar_4 = most_similar_4[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_5 = torch.topk(cosine_sim_5, k=2, dim=1,largest=True)
                most_similar_5 = most_similar_5[:, 0]  # 获取除自己外最相似的索引
        else:
            if x_t_re.shape[0] == 1:
                _, most_similar_1 = torch.topk(cosine_sim_1, k=1, dim=1,largest=True)
                most_similar_1_temp = most_similar_1[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_4 = torch.topk(cosine_sim_4, k=1, dim=1,largest=True)
                most_similar_4 = most_similar_4[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_5 = torch.topk(cosine_sim_5, k=1, dim=1,largest=True)
                most_similar_5 = most_similar_5[:, 0]  # 获取除自己外最相似的索引
            else:
                _, most_similar_1 = torch.topk(cosine_sim_1, k=2, dim=1,largest=True)
                most_similar_1_temp = most_similar_1[:, 0]  # 获取除自己外最相似的索引
                most_similar_1_temp[label] = most_similar_1[label,1]
                _, most_similar_4 = torch.topk(cosine_sim_4, k=2, dim=1,largest=True)
                most_similar_4 = most_similar_4[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_5 = torch.topk(cosine_sim_5, k=2, dim=1,largest=True)
                most_similar_5 = most_similar_5[:, 0]  # 获取除自己外最相似的索引

        cost_t = x_t_re[most_similar_1_temp]
        cost_a = x_a_re[most_similar_4]
        cost_v = x_v_re[most_similar_5]
        #cos_t_cat = torch.stack((cost_t, cost_a, cost_v), dim=1)

        if label is None :
            if x_t_re.shape[0] == 1:
                _, most_similar_2 = torch.topk(cosine_sim_2, k=1, dim=1)
                most_similar_2_temp = most_similar_2[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_6 = torch.topk(cosine_sim_6, k=1, dim=1,largest=True)
                most_similar_6 = most_similar_6[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_7 = torch.topk(cosine_sim_7, k=1, dim=1,largest=True)
                most_similar_7 = most_similar_7[:, 0]  # 获取除自己外最相似的索引
            else:
                _, most_similar_2 = torch.topk(cosine_sim_2, k=2, dim=1)
                most_similar_2_temp = most_similar_2[:, 1]  # 获取除自己外最相似的索引
                _, most_similar_6 = torch.topk(cosine_sim_6, k=2, dim=1,largest=True)
                most_similar_6 = most_similar_6[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_7 = torch.topk(cosine_sim_7, k=2, dim=1,largest=True)
                most_similar_7 = most_similar_7[:, 0]  # 获取除自己外最相似的索引
        else:
            if x_t_re.shape[0] == 1:
                _, most_similar_2 = torch.topk(cosine_sim_2, k=1, dim=1)
                most_similar_2_temp = most_similar_2[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_6 = torch.topk(cosine_sim_6, k=1, dim=1,largest=True)
                most_similar_6 = most_similar_6[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_7 = torch.topk(cosine_sim_7, k=1, dim=1,largest=True)
                most_similar_7 = most_similar_7[:, 0]  # 获取除自己外最相似的索引
            else :
                _, most_similar_2 = torch.topk(cosine_sim_2, k=2, dim=1)
                most_similar_2_temp = most_similar_2[:, 0]  # 获取除自己外最相似的索引
                most_similar_2_temp[label] = most_similar_2[label,1]
                _, most_similar_6 = torch.topk(cosine_sim_6, k=2, dim=1,largest=True)
                most_similar_6 = most_similar_6[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_7 = torch.topk(cosine_sim_7, k=2, dim=1,largest=True)
                most_similar_7 = most_similar_7[:, 0]  # 获取除自己外最相似的索引

        cosa_a = x_a_re[most_similar_2_temp]
        cosa_t = x_t_re[most_similar_6]
        cosa_v = x_v_re[most_similar_7]
        #cos_a_cat = torch.stack((cosa_a, cosa_t, cosa_v), dim=1)
        
        if label is None :
            if x_t_re.shape[0] == 1:
                _, most_similar_3 = torch.topk(cosine_sim_3, k=1, dim=1)
                most_similar_3_temp = most_similar_3[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_8 = torch.topk(cosine_sim_8, k=1, dim=1,largest=True)
                most_similar_8 = most_similar_8[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_9 = torch.topk(cosine_sim_9, k=1, dim=1,largest=True)
                most_similar_9 = most_similar_9[:, 0]  # 获取除自己外最相似的索引
            else :
                _, most_similar_3 = torch.topk(cosine_sim_3, k=2, dim=1)
                most_similar_3_temp = most_similar_3[:, 1]  # 获取除自己外最相似的索引
                _, most_similar_8 = torch.topk(cosine_sim_8, k=2, dim=1,largest=True)
                most_similar_8 = most_similar_8[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_9 = torch.topk(cosine_sim_9, k=2, dim=1,largest=True)
                most_similar_9 = most_similar_9[:, 0]  # 获取除自己外最相似的索引
        else:
            if x_t_re.shape[0] == 1:
                _, most_similar_3 = torch.topk(cosine_sim_3, k=1, dim=1)
                most_similar_3_temp = most_similar_3[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_8 = torch.topk(cosine_sim_8, k=1, dim=1,largest=True)
                most_similar_8 = most_similar_8[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_9 = torch.topk(cosine_sim_9, k=1, dim=1,largest=True)
                most_similar_9 = most_similar_9[:, 0]  # 获取除自己外最相似的索引
            else :
                _, most_similar_3 = torch.topk(cosine_sim_3, k=2, dim=1)
                most_similar_3_temp = most_similar_3[:, 0]  # 获取除自己外最相似的索引
                most_similar_3_temp[label] = most_similar_3[label,1]
                _, most_similar_8 = torch.topk(cosine_sim_8, k=2, dim=1,largest=True)
                most_similar_8 = most_similar_8[:, 0]  # 获取除自己外最相似的索引
                _, most_similar_9 = torch.topk(cosine_sim_9, k=2, dim=1,largest=True)
                most_similar_9 = most_similar_9[:, 0]  # 获取除自己外最相似的索引

        cosv_v = x_v_re[most_similar_3_temp]
        cosv_t = x_t_re[most_similar_8]
        cosv_a = x_a_re[most_similar_9]
        #cos_v_cat = torch.stack((cosv_v, cosv_t, cosv_a), dim=1)

        # cos_t_cat = torch.stack((cost_t, cosa_t, cosv_t), dim=0)
        # cos_a_cat = torch.stack((cost_a, cosa_a, cosv_a), dim=0)
        # cos_v_cat = torch.stack((cost_v, cosa_v, cosv_v), dim=0)

        cos_t_cat = torch.stack((cost_t, cost_a, cost_v), dim=0)
        cos_a_cat = torch.stack((cosa_a, cosa_t, cosa_v), dim=0)
        cos_v_cat = torch.stack((cosv_v, cosv_t, cosv_a), dim=0)
        
        # t_index = torch.stack((most_similar_1_temp, most_similar_4, most_similar_5), dim=0)
        # a_index = torch.stack((most_similar_2_temp, most_similar_6, most_similar_7), dim=0)
        # v_index = torch.stack((most_similar_3_temp, most_similar_8, most_similar_9), dim=0)
        sim = torch.stack([cosine_sim_1,cosine_sim_4,cosine_sim_5,cosine_sim_2,cosine_sim_6,cosine_sim_7,cosine_sim_3,cosine_sim_8,cosine_sim_9],dim = 0)
        
        return cos_t_cat,cos_a_cat,cos_v_cat,sim
        # return t_index,a_index,v_index
    

    def get_sample_data(self,cat_t,cat_a,cat_v):
        #get text modality prompt
        x_l = torch.cat(
            [self.sample_prompt[0, :, :], self.cat02t(cat_t[0].unsqueeze(0))[0],self.cat12t(cat_t[1].unsqueeze(0))[0],self.cat22t(cat_t[2].unsqueeze(0))[0]],
            # [self.sample_prompt[0, :, :], cat_t[0].unsqueeze(0)[0],cat_t[1].unsqueeze(0)[0],cat_t[2].unsqueeze(0)[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_l = self.t_tavp(x_l.transpose(1, 2)).transpose(1, 2) 
        x_l = self.t_tavp(x_l).transpose(1, 2) 

        #get audio modality prompt
        x_a = torch.cat(
            [self.sample_prompt[1, :, :], self.cat02a(cat_a[0].unsqueeze(0))[0],self.cat12a(cat_a[1].unsqueeze(0))[0],self.cat22a(cat_a[2].unsqueeze(0))[0]],
            # [self.sample_prompt[1, :, :], cat_a[0].unsqueeze(0)[0],cat_a[1].unsqueeze(0)[0],cat_a[2].unsqueeze(0)[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_a = self.a_tavp(x_a.transpose(1, 2)).transpose(1, 2) 
        x_a = self.a_tavp(x_a).transpose(1, 2) 

        #get visual modality prompt
        x_v = torch.cat(
            [self.sample_prompt[2, :, :], self.cat02v(cat_v[0].unsqueeze(0))[0],self.cat12v(cat_v[1].unsqueeze(0))[0],self.cat22v(cat_v[2].unsqueeze(0))[0]],
            # [self.sample_prompt[2, :, :], cat_v[0].unsqueeze(0)[0],cat_v[1].unsqueeze(0)[0],cat_v[2].unsqueeze(0)[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_v = self.v_tavp(x_v.transpose(1, 2)).transpose(1, 2) 
        x_v = self.v_tavp(x_v).transpose(1, 2) 

        return x_l, x_a, x_v
    
    

    def forward(self, text, audio, visual, label_ids=None, epoch=0):
        visual = self.v_convert_layer(visual)
        audio = self.a_convert_layer(audio)
        text = self.t_convert_layer(text)
        
        if label_ids is None :
            cos_t_cat,cos_a_cat,cos_v_cat ,sim_p = self.cal_cos_similarity(text,audio,visual,text,audio,visual)
            # visual = self.v_mlp_layer(visual)
            # audio = self.a_mlp_layer(audio)
            # text = self.t_mlp_layer(text)
            # cos_t_cat = self.t_p_layer(cos_t_cat)
            # cos_a_cat = self.a_p_layer(cos_a_cat)
            # cos_v_cat = self.v_p_layer(cos_v_cat)
            # t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
            # a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
            # v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
        # else :
        #     label_ids = label_ids.squeeze()
        #     if len(label_ids.shape) == 0 :
        #         label_ids = label_ids.unsqueeze(0)
        #     positive_label = torch.nonzero(label_ids >= 0).squeeze()
        #     negative_label = torch.nonzero(label_ids < 0).squeeze()
        #     if len(positive_label.shape) == 0 :
        #         positive_label = positive_label.unsqueeze(0)
        #     if len(negative_label.shape) == 0 :
        #         negative_label = negative_label.unsqueeze(0)
        #     positive_text,positive_visual,positive_audio = text[positive_label],visual[positive_label],audio[positive_label]
        #     negative_text,negative_visual,negative_audio = text[negative_label],visual[negative_label],audio[negative_label]
        #     if positive_label.numel() == 0:
        #         cos_t_cat,cos_a_cat,cos_v_cat,sim_n = self.cal_cos_similarity(text,audio,visual,negative_text,negative_audio,negative_visual,negative_label)
        #         # visual = self.v_mlp_layer(visual)
        #         # audio = self.a_mlp_layer(audio)
        #         # text = self.t_mlp_layer(text)
        #         # cos_t_cat = self.t_p_layer(cos_t_cat)
        #         # cos_a_cat = self.a_p_layer(cos_a_cat)
        #         # cos_v_cat = self.v_p_layer(cos_v_cat)
        #         t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
        #         a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
        #         v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
        #     elif negative_label.numel() == 0:
        #         cos_t_cat,cos_a_cat,cos_v_cat ,sim_p= self.cal_cos_similarity(text,audio,visual,positive_text,positive_audio,positive_visual,positive_label)
        #         # visual = self.v_mlp_layer(visual)
        #         # audio = self.a_mlp_layer(audio)
        #         # text = self.t_mlp_layer(text)
        #         # cos_t_cat = self.t_p_layer(cos_t_cat)
        #         # cos_a_cat = self.a_p_layer(cos_a_cat)
        #         # cos_v_cat = self.v_p_layer(cos_v_cat)
        #         t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
        #         a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
        #         v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
        #     else:
        #         cos_t_cat,cos_a_cat,cos_v_cat ,sim_p = self.cal_cos_similarity(text,audio,visual,positive_text,positive_audio,positive_visual,positive_label)
        #         cos_t_cat_n,cos_a_cat_n,cos_v_cat_n ,sim_n = self.cal_cos_similarity(text,audio,visual,negative_text,negative_audio,negative_visual,negative_label)
        #         cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n = self.find_features(cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n,label_ids)
                
                # visual = self.v_mlp_layer(visual)
                # audio = self.a_mlp_layer(audio)
                # text = self.t_mlp_layer(text)
                # cos_t_cat = self.t_p_layer(cos_t_cat)
                # cos_a_cat = self.a_p_layer(cos_a_cat)
                # cos_v_cat = self.v_p_layer(cos_v_cat)
                # cos_t_cat_n = self.t_n_layer(cos_t_cat_n)
                # cos_a_cat_n = self.a_n_layer(cos_a_cat_n)
                # cos_v_cat_n = self.v_n_layer(cos_v_cat_n)
                # t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])+self.cal_l2_loss2(text,cos_t_cat_n[0],cos_t_cat_n[1],cos_t_cat_n[2],0)
                # a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])+self.cal_l2_loss2(audio,cos_a_cat_n[1],cos_a_cat_n[0],cos_a_cat_n[2],1)
                # v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])+self.cal_l2_loss2(visual,cos_v_cat_n[1],cos_v_cat_n[2],cos_v_cat_n[0],2)
            

        # print(cos_t_cat.size())
        # get sample data
        cos_t_cat = cos_t_cat.permute(1,0,3,2)
        cos_a_cat = cos_a_cat.permute(1,0,3,2)
        cos_v_cat = cos_v_cat.permute(1,0,3,2)
        t_s_p, a_s_p, v_s_p = None, None, None
        for idx in range(len(text)):
            x_l_temp, x_a_temp, x_v_temp = self.get_sample_data(
                cos_t_cat[idx],cos_a_cat[idx],cos_v_cat[idx]
            )
            if t_s_p is None:
                t_s_p = x_l_temp
                a_s_p = x_a_temp
                v_s_p = x_v_temp
            else:
                t_s_p = torch.cat([t_s_p, x_l_temp], dim=0)
                a_s_p = torch.cat([a_s_p, x_a_temp], dim=0)
                v_s_p = torch.cat([v_s_p, x_v_temp], dim=0)
        # t_s_p = t_s_p.transpose(1,2)
        # a_s_p = a_s_p.transpose(1,2)
        # v_s_p = v_s_p.transpose(1,2)
        cos_t_cat = cos_t_cat.permute(1,0,3,2)
        cos_a_cat = cos_a_cat.permute(1,0,3,2)
        cos_v_cat = cos_v_cat.permute(1,0,3,2)
        
        
        
        #get modality data
        text = text.transpose(1,2)
        visual = visual.transpose(1, 2)
        audio = audio.transpose(1, 2)
        t_m_p, a_m_p, v_m_p = None, None, None
        for idx in range(len(text)):
            x_l_temp, x_a_temp, x_v_temp = self.get_modality_data(
                text[idx], audio[idx], visual[idx]
            )
            if t_m_p is None:
                t_m_p = x_l_temp
                a_m_p = x_a_temp
                v_m_p = x_v_temp
            else:
                t_m_p = torch.cat([t_m_p, x_l_temp], dim=0)
                a_m_p = torch.cat([a_m_p, x_a_temp], dim=0)
                v_m_p = torch.cat([v_m_p, x_v_temp], dim=0)
        # t_m_p = t_m_p.transpose(1,2)
        # a_m_p = a_m_p.transpose(1,2)
        # v_m_p = v_m_p.transpose(1,2)
        text = text.transpose(1,2)
        visual = visual.transpose(1, 2)
        audio = audio.transpose(1, 2)

        # # print(text.size())
        # # print(t_m_p.size())
        # # print(t_s_p.size())



        #Cross-attention Block
        text = text.permute(1,0,2)
        visual = visual.permute(1,0,2)
        audio = audio.permute(1,0,2)

        t_m_p = t_m_p.permute(1,0,2)
        a_m_p = a_m_p.permute(1,0,2)
        v_m_p = v_m_p.permute(1,0,2)
        # # print(t_m_p.size())

        t_s_p = t_s_p.permute(1,0,2)
        a_s_p = a_s_p.permute(1,0,2)
        v_s_p = v_s_p.permute(1,0,2)

        t_s_output = self.t_s_encoder(t_s_p,text)
        t_m_output = self.t_m_encoder(t_m_p,text)
        a_s_output = self.a_s_encoder(a_s_p,audio)
        a_m_output = self.a_m_encoder(a_m_p,audio)
        v_s_output = self.v_s_encoder(v_s_p,visual)
        v_m_output = self.v_m_encoder(v_m_p,visual)
        # print(t_s_output.size())

        # t_s_output = self.t_s_encoder(t_s_p,text,tgt_key_padding_mask = text_mask)
        # t_m_output = self.t_m_encoder(t_m_p,text,tgt_key_padding_mask = text_mask)
        # a_s_output = self.a_s_encoder(a_s_p,audio,tgt_key_padding_mask = audio_mask)
        # a_m_output = self.a_m_encoder(a_m_p,audio,tgt_key_padding_mask = audio_mask)
        # v_s_output = self.v_s_encoder(v_s_p,visual,tgt_key_padding_mask = visual_mask)
        # v_m_output = self.v_m_encoder(v_m_p,visual,tgt_key_padding_mask = visual_mask)

        t_output = torch.cat([t_m_output, t_s_output], dim=2)
        a_output = torch.cat([a_m_output, a_s_output], dim=2)
        v_output = torch.cat([v_m_output, v_s_output], dim=2)
        # # print(t_output.size())

        t_output = self.t_encoder(t_output)
        a_output = self.a_encoder(a_output)
        v_output = self.v_encoder(v_output)

        # t_output = self.t_encoder(text,src_key_padding_mask = text_mask)
        # a_output = self.a_encoder(audio,src_key_padding_mask = audio_mask)
        # v_output = self.v_encoder(visual,src_key_padding_mask = visual_mask)
        # print(t_output.size())

        # outputs = torch.cat([text_dg[0],visual_dg[0],audio_dg[0]],dim = 1)
        # outputs = torch.stack((t_s_output,t_m_output,a_s_output,a_m_output,v_s_output,v_m_output), dim=1)
        # outputs = t_s_output+t_m_output+a_s_output+a_m_output+v_s_output+v_m_output
        outputs = torch.cat([t_output[0],a_output[0],v_output[0]],dim = 1)
        # outputs = torch.cat((t_s_output,t_m_output,a_s_output,a_m_output,v_s_output,v_m_output), dim=1)
        # outputs = torch.stack((t_s_output,t_m_output,a_s_output,a_m_output), dim=1)
        # outputs = outputs.permute(1,0,2)
        # outputs = self.encoder(outputs)#去掉
        # outputs = outputs[0]
        
        # print(outputs.size())
        logits = self.classifier(outputs)
        # logits = self.classifier(outputs[0])

        # text = text.permute(1, 0 ,2)
        # text_dg = self.tdg_encoder(text, text)

        # visual = visual.permute(1, 0, 2)
        # visual_dg = self.vdg_encoder(visual, visual)
        
        if logits.shape[0] == 1:
            return logits.squeeze().unsqueeze(0)
        else:
            return logits.squeeze()
        # return logits.squeeze()
class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        # if is_Fusion:
        #     self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        # else:
        #     self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        # self.act = nn.GELU()
        if is_Fusion:
            self.mlp = nn.Linear(dim, embed_dim)
        else:
            self.mlp = nn.Linear(dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # return self.act(self.conv(x))
        x = x.transpose(1,2)
        x = self.mlp(x)
        x = x.transpose(1,2)
        return x

class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.prompt_dim*6 ,args.prompt_dim*6)
        self.dropout = nn.Dropout(p=args.classifier_dropout)
        self.out_proj = nn.Linear(args.prompt_dim*6, args.ds_label_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class NORAGModel(nn.Module):
    def __init__(self, args):
        super(NORAGModel, self).__init__()
            
        self.v_convert_layer = nn.Linear(args.v_dim, args.prompt_dim)
        self.a_convert_layer = nn.Linear(args.a_dim, args.prompt_dim)
        self.t_convert_layer = nn.Linear(args.t_dim, args.prompt_dim)

        self.l2a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.l2v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.l2l = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.v2a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.v2l = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.v2v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.a2v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.a2l = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.a2a = MLPLayer(args.prompt_dim, args.prompt_dim)

        self.cat02t = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat12t = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat22t = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat02a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat12a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat22a = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat02v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat12v = MLPLayer(args.prompt_dim, args.prompt_dim)
        self.cat22v = MLPLayer(args.prompt_dim, args.prompt_dim)

        self.a_lvp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.v_alp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.l_avp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )


        self.t_tavp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.a_tavp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.v_tavp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )


        # self.modality_prompt = nn.Parameter(torch.randn((3,args.prompt_dim,args.prompt_len)))
        self.modality_prompt = nn.Parameter(
            torch.empty((3,args.prompt_dim,args.prompt_len))
        )
        for prompt in self.modality_prompt:
            init.kaiming_uniform_(prompt, a=math.sqrt(5))
        
        self.sample_prompt = nn.Parameter(
            torch.empty((3,args.prompt_dim,args.prompt_len))
        )
        for prompt in self.sample_prompt:
            init.kaiming_uniform_(prompt, a=math.sqrt(5))
        self.t_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.t_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.a_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.a_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.v_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.v_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)

        t_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim*2,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.t_encoder = nn.TransformerEncoder(
            encoder_layer=t_encoder_layer,
            num_layers=args.layers
        )
        a_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim*2,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=a_encoder_layer,
            num_layers=args.layers
        )
        v_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim*2,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=v_encoder_layer,
            num_layers=args.layers
        )

        self.classifier = T5ClassificationHead(args)

        # self.t_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.a_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.v_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)

        # self.t_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.a_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.v_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)

        # self.t_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.a_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        # self.v_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)


    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def get_modality_data(self, x_l, x_a, x_v):
        x_l, x_a, x_v = x_l.unsqueeze(dim=0), x_a.unsqueeze(dim=0), x_v.unsqueeze(dim=0)

        #get text modality prompt
        x_l_p = torch.cat(
            [self.modality_prompt[0, :, :], self.l2l(x_l)[0] ,self.a2l(x_a)[0], self.v2l(x_v)[0]],
            # [self.modality_prompt[0, :, :], x_l[0] ,x_a[0], x_v[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_l_p = self.l_avp(x_l_p.transpose(1, 2)).transpose(1, 2) 
        x_l_p = self.l_avp(x_l_p).transpose(1, 2) 

        #get audio modality prompt
        x_a_p = torch.cat(
            [self.modality_prompt[1, :, :], self.a2a(x_a)[0],self.l2a(x_l)[0], self.v2a(x_v)[0]],
            # [self.modality_prompt[1, :, :], x_a[0],x_l[0], x_v[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_a_p = self.a_lvp(x_a_p.transpose(1, 2)).transpose(1, 2) 
        x_a_p = self.a_lvp(x_a_p).transpose(1, 2) 

        #get visual modality prompt
        x_v_p = torch.cat(
            [self.modality_prompt[2, :, :], self.v2v(x_v)[0], self.l2v(x_l)[0], self.a2v(x_a)[0]],
            # [self.modality_prompt[2, :, :], x_v[0], x_l[0], x_a[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_v_p = self.v_alp(x_v_p.transpose(1, 2)).transpose(1, 2) 
        x_v_p = self.v_alp(x_v_p).transpose(1, 2) 

        return x_l_p, x_a_p, x_v_p

    def pad_tensor(self,tensor, target_dim):
        padding_size = target_dim - tensor.size(1)
        if padding_size > 0:
            padding = torch.zeros((tensor.size(0), padding_size), dtype=tensor.dtype).to(DEVICE)
            tensor = torch.cat((tensor, padding), dim=1)
        return tensor

    def cal_cos_similarity(self,x_t,x_a,x_v,x_t_re,x_a_re,x_v_re,label = None):
        """
        :param x_t: Shape (batch_size, sequence_length, feature_dim)
        :param x_a: Shape (batch_size, sequence_length, feature_dim)
        :param x_v: Shape (batch_size, sequence_length, feature_dim)
        :return: tuple of tensors containing the most similar samples for each modality
        """
        if len(x_t_re.shape) < 3:
            x_t_re = x_t_re.unsqueeze(0)
            x_a_re = x_a_re.unsqueeze(0)
            x_v_re = x_v_re.unsqueeze(0)
        
    

        x_t_flat = x_t.reshape(x_t.size(0), x_t.size(1)*x_t.size(2))  # (batch_size, seq_len * feature_dim)
        x_a_flat = x_a.reshape(x_a.size(0), x_a.size(1)*x_a.size(2))  # (batch_size, seq_len * feature_dim)
        x_v_flat = x_v.reshape(x_v.size(0), x_v.size(1)*x_v.size(2))  # (batch_size, seq_len * feature_dim)

        max_dim = max(x_t_flat.shape[1],x_a_flat.shape[1],x_v_flat.shape[1])
        x_t_flat = self.pad_tensor(x_t_flat,max_dim)
        x_a_flat = self.pad_tensor(x_a_flat,max_dim)
        x_v_flat = self.pad_tensor(x_v_flat,max_dim)

        x_t_re_flat = x_t_re.reshape(x_t_re.size(0), x_t_re.size(1)*x_t_re.size(2))  # (batch_size, seq_len * feature_dim)
        x_a_re_flat = x_a_re.reshape(x_a_re.size(0), x_a_re.size(1)*x_a_re.size(2))  # (batch_size, seq_len * feature_dim)
        x_v_re_flat = x_v_re.reshape(x_v_re.size(0), x_v_re.size(1)*x_v_re.size(2))  # (batch_size, seq_len * feature_dim)

        max_dim = max(x_t_re_flat.shape[1],x_a_re_flat.shape[1],x_v_re_flat.shape[1])
        x_t_re_flat = self.pad_tensor(x_t_re_flat,max_dim)
        x_a_re_flat = self.pad_tensor(x_a_re_flat,max_dim)
        x_v_re_flat = self.pad_tensor(x_v_re_flat,max_dim)
        
        # 2. 计算每个模态的余弦相似度
        # cosine_sim_1 = F.cosine_similarity(x_t_flat.unsqueeze(1), x_t_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        # cosine_sim_2 = F.cosine_similarity(x_a_flat.unsqueeze(1), x_a_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        # cosine_sim_3 = F.cosine_similarity(x_v_flat.unsqueeze(1), x_v_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        # cosine_sim_4 = F.cosine_similarity(x_t_flat.unsqueeze(1), x_a_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        # cosine_sim_5 = F.cosine_similarity(x_t_flat.unsqueeze(1), x_v_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        # cosine_sim_6 = F.cosine_similarity(x_a_flat.unsqueeze(1), x_t_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        # cosine_sim_7 = F.cosine_similarity(x_a_flat.unsqueeze(1), x_v_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        # cosine_sim_8 = F.cosine_similarity(x_v_flat.unsqueeze(1), x_t_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        # cosine_sim_9 = F.cosine_similarity(x_v_flat.unsqueeze(1), x_a_re_flat.unsqueeze(0), dim=-1)  # (batch_size, batch_size)
        
        # if label is None :
        #     if x_t_re.shape[0] == 1:
        #         _, most_similar_1 = torch.topk(cosine_sim_1, k=1, dim=1,largest=True)
        #         most_similar_1_temp = most_similar_1[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_4 = torch.topk(cosine_sim_4, k=1, dim=1,largest=True)
        #         most_similar_4 = most_similar_4[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_5 = torch.topk(cosine_sim_5, k=1, dim=1,largest=True)
        #         most_similar_5 = most_similar_5[:, 0]  # 获取除自己外最相似的索引
        #     else:
        #         _, most_similar_1 = torch.topk(cosine_sim_1, k=2, dim=1,largest=True)
        #         most_similar_1_temp = most_similar_1[:, 1]  # 获取除自己外最相似的索引
        #         _, most_similar_4 = torch.topk(cosine_sim_4, k=2, dim=1,largest=True)
        #         most_similar_4 = most_similar_4[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_5 = torch.topk(cosine_sim_5, k=2, dim=1,largest=True)
        #         most_similar_5 = most_similar_5[:, 0]  # 获取除自己外最相似的索引
        # else:
        #     if x_t_re.shape[0] == 1:
        #         _, most_similar_1 = torch.topk(cosine_sim_1, k=1, dim=1,largest=True)
        #         most_similar_1_temp = most_similar_1[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_4 = torch.topk(cosine_sim_4, k=1, dim=1,largest=True)
        #         most_similar_4 = most_similar_4[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_5 = torch.topk(cosine_sim_5, k=1, dim=1,largest=True)
        #         most_similar_5 = most_similar_5[:, 0]  # 获取除自己外最相似的索引
        #     else:
        #         _, most_similar_1 = torch.topk(cosine_sim_1, k=2, dim=1,largest=True)
        #         most_similar_1_temp = most_similar_1[:, 0]  # 获取除自己外最相似的索引
        #         most_similar_1_temp[label] = most_similar_1[label,1]
        #         _, most_similar_4 = torch.topk(cosine_sim_4, k=2, dim=1,largest=True)
        #         most_similar_4 = most_similar_4[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_5 = torch.topk(cosine_sim_5, k=2, dim=1,largest=True)
        #         most_similar_5 = most_similar_5[:, 0]  # 获取除自己外最相似的索引

        # cost_t = x_t_re[most_similar_1_temp]
        # cost_a = x_a_re[most_similar_4]
        # cost_v = x_v_re[most_similar_5]
        #cos_t_cat = torch.stack((cost_t, cost_a, cost_v), dim=1)

        # if label is None :
        #     if x_t_re.shape[0] == 1:
        #         _, most_similar_2 = torch.topk(cosine_sim_2, k=1, dim=1)
        #         most_similar_2_temp = most_similar_2[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_6 = torch.topk(cosine_sim_6, k=1, dim=1,largest=True)
        #         most_similar_6 = most_similar_6[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_7 = torch.topk(cosine_sim_7, k=1, dim=1,largest=True)
        #         most_similar_7 = most_similar_7[:, 0]  # 获取除自己外最相似的索引
        #     else:
        #         _, most_similar_2 = torch.topk(cosine_sim_2, k=2, dim=1)
        #         most_similar_2_temp = most_similar_2[:, 1]  # 获取除自己外最相似的索引
        #         _, most_similar_6 = torch.topk(cosine_sim_6, k=2, dim=1,largest=True)
        #         most_similar_6 = most_similar_6[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_7 = torch.topk(cosine_sim_7, k=2, dim=1,largest=True)
        #         most_similar_7 = most_similar_7[:, 0]  # 获取除自己外最相似的索引
        # else:
        #     if x_t_re.shape[0] == 1:
        #         _, most_similar_2 = torch.topk(cosine_sim_2, k=1, dim=1)
        #         most_similar_2_temp = most_similar_2[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_6 = torch.topk(cosine_sim_6, k=1, dim=1,largest=True)
        #         most_similar_6 = most_similar_6[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_7 = torch.topk(cosine_sim_7, k=1, dim=1,largest=True)
        #         most_similar_7 = most_similar_7[:, 0]  # 获取除自己外最相似的索引
        #     else :
        #         _, most_similar_2 = torch.topk(cosine_sim_2, k=2, dim=1)
        #         most_similar_2_temp = most_similar_2[:, 0]  # 获取除自己外最相似的索引
        #         most_similar_2_temp[label] = most_similar_2[label,1]
        #         _, most_similar_6 = torch.topk(cosine_sim_6, k=2, dim=1,largest=True)
        #         most_similar_6 = most_similar_6[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_7 = torch.topk(cosine_sim_7, k=2, dim=1,largest=True)
        #         most_similar_7 = most_similar_7[:, 0]  # 获取除自己外最相似的索引

        # cosa_a = x_a_re[most_similar_2_temp]
        # cosa_t = x_t_re[most_similar_6]
        # cosa_v = x_v_re[most_similar_7]
        # #cos_a_cat = torch.stack((cosa_a, cosa_t, cosa_v), dim=1)
        
        # if label is None :
        #     if x_t_re.shape[0] == 1:
        #         _, most_similar_3 = torch.topk(cosine_sim_3, k=1, dim=1)
        #         most_similar_3_temp = most_similar_3[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_8 = torch.topk(cosine_sim_8, k=1, dim=1,largest=True)
        #         most_similar_8 = most_similar_8[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_9 = torch.topk(cosine_sim_9, k=1, dim=1,largest=True)
        #         most_similar_9 = most_similar_9[:, 0]  # 获取除自己外最相似的索引
        #     else :
        #         _, most_similar_3 = torch.topk(cosine_sim_3, k=2, dim=1)
        #         most_similar_3_temp = most_similar_3[:, 1]  # 获取除自己外最相似的索引
        #         _, most_similar_8 = torch.topk(cosine_sim_8, k=2, dim=1,largest=True)
        #         most_similar_8 = most_similar_8[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_9 = torch.topk(cosine_sim_9, k=2, dim=1,largest=True)
        #         most_similar_9 = most_similar_9[:, 0]  # 获取除自己外最相似的索引
        # else:
        #     if x_t_re.shape[0] == 1:
        #         _, most_similar_3 = torch.topk(cosine_sim_3, k=1, dim=1)
        #         most_similar_3_temp = most_similar_3[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_8 = torch.topk(cosine_sim_8, k=1, dim=1,largest=True)
        #         most_similar_8 = most_similar_8[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_9 = torch.topk(cosine_sim_9, k=1, dim=1,largest=True)
        #         most_similar_9 = most_similar_9[:, 0]  # 获取除自己外最相似的索引
        #     else :
        #         _, most_similar_3 = torch.topk(cosine_sim_3, k=2, dim=1)
        #         most_similar_3_temp = most_similar_3[:, 0]  # 获取除自己外最相似的索引
        #         most_similar_3_temp[label] = most_similar_3[label,1]
        #         _, most_similar_8 = torch.topk(cosine_sim_8, k=2, dim=1,largest=True)
        #         most_similar_8 = most_similar_8[:, 0]  # 获取除自己外最相似的索引
        #         _, most_similar_9 = torch.topk(cosine_sim_9, k=2, dim=1,largest=True)
        #         most_similar_9 = most_similar_9[:, 0]  # 获取除自己外最相似的索引

        # cosv_v = x_v_re[most_similar_3_temp]
        # cosv_t = x_t_re[most_similar_8]
        # cosv_a = x_a_re[most_similar_9]
        #cos_v_cat = torch.stack((cosv_v, cosv_t, cosv_a), dim=1)

        # cos_t_cat = torch.stack((cost_t, cosa_t, cosv_t), dim=0)
        # cos_a_cat = torch.stack((cost_a, cosa_a, cosv_a), dim=0)
        # cos_v_cat = torch.stack((cost_v, cosa_v, cosv_v), dim=0)

        # cos_t_cat = torch.stack((cost_t, cost_a, cost_v), dim=0)
        # cos_a_cat = torch.stack((cosa_a, cosa_t, cosa_v), dim=0)
        # cos_v_cat = torch.stack((cosv_v, cosv_t, cosv_a), dim=0)
        
        # t_index = torch.stack((most_similar_1_temp, most_similar_4, most_similar_5), dim=0)
        # a_index = torch.stack((most_similar_2_temp, most_similar_6, most_similar_7), dim=0)
        # v_index = torch.stack((most_similar_3_temp, most_similar_8, most_similar_9), dim=0)
        # sim = torch.stack([cosine_sim_1,cosine_sim_4,cosine_sim_5,cosine_sim_2,cosine_sim_6,cosine_sim_7,cosine_sim_3,cosine_sim_8,cosine_sim_9],dim = 0)
        
        # return cos_t_cat,cos_a_cat,cos_v_cat,sim
        # return t_index,a_index,v_index
        return 

    def get_sample_data(self,cat_t,cat_a,cat_v):
        #get text modality prompt
        x_l = torch.cat(
            [self.sample_prompt[0, :, :], self.cat02t(cat_t[0].unsqueeze(0))[0],self.cat12t(cat_t[1].unsqueeze(0))[0],self.cat22t(cat_t[2].unsqueeze(0))[0]],
            # [self.sample_prompt[0, :, :], cat_t[0].unsqueeze(0)[0],cat_t[1].unsqueeze(0)[0],cat_t[2].unsqueeze(0)[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_l = self.t_tavp(x_l.transpose(1, 2)).transpose(1, 2) 
        x_l = self.t_tavp(x_l).transpose(1, 2) 

        #get audio modality prompt
        x_a = torch.cat(
            [self.sample_prompt[1, :, :], self.cat02a(cat_a[0].unsqueeze(0))[0],self.cat12a(cat_a[1].unsqueeze(0))[0],self.cat22a(cat_a[2].unsqueeze(0))[0]],
            # [self.sample_prompt[1, :, :], cat_a[0].unsqueeze(0)[0],cat_a[1].unsqueeze(0)[0],cat_a[2].unsqueeze(0)[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_a = self.a_tavp(x_a.transpose(1, 2)).transpose(1, 2) 
        x_a = self.a_tavp(x_a).transpose(1, 2) 

        #get visual modality prompt
        x_v = torch.cat(
            [self.sample_prompt[2, :, :], self.cat02v(cat_v[0].unsqueeze(0))[0],self.cat12v(cat_v[1].unsqueeze(0))[0],self.cat22v(cat_v[2].unsqueeze(0))[0]],
            # [self.sample_prompt[2, :, :], cat_v[0].unsqueeze(0)[0],cat_v[1].unsqueeze(0)[0],cat_v[2].unsqueeze(0)[0]],
            dim=0,
        ).unsqueeze(dim=0)
        # x_v = self.v_tavp(x_v.transpose(1, 2)).transpose(1, 2) 
        x_v = self.v_tavp(x_v).transpose(1, 2) 

        return x_l, x_a, x_v


    def forward(self, text, audio, visual, label_ids=None, epoch=0):
        visual = self.v_convert_layer(visual)
        audio = self.a_convert_layer(audio)
        text = self.t_convert_layer(text)
        
        t_loss,a_loss,v_loss = None,None,None
        if label_ids is None :
            # cos_t_cat,cos_a_cat,cos_v_cat ,sim_p = self.cal_cos_similarity(text,audio,visual,text,audio,visual)
            _ = self.cal_cos_similarity(text,audio,visual,text,audio,visual)
            # visual = self.v_mlp_layer(visual)
            # audio = self.a_mlp_layer(audio)
            # text = self.t_mlp_layer(text)
            # cos_t_cat = self.t_p_layer(cos_t_cat)
            # cos_a_cat = self.a_p_layer(cos_a_cat)
            # cos_v_cat = self.v_p_layer(cos_v_cat)
            # t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
            # a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
            # v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
        # else :
        #     label_ids = label_ids.squeeze()
        #     if len(label_ids.shape) == 0 :
        #         label_ids = label_ids.unsqueeze(0)
        #     positive_label = torch.nonzero(label_ids >= 0).squeeze()
        #     negative_label = torch.nonzero(label_ids < 0).squeeze()
        #     if len(positive_label.shape) == 0 :
        #         positive_label = positive_label.unsqueeze(0)
        #     if len(negative_label.shape) == 0 :
        #         negative_label = negative_label.unsqueeze(0)
        #     positive_text,positive_visual,positive_audio = text[positive_label],visual[positive_label],audio[positive_label]
        #     negative_text,negative_visual,negative_audio = text[negative_label],visual[negative_label],audio[negative_label]
        #     if positive_label.numel() == 0:
        #         cos_t_cat,cos_a_cat,cos_v_cat,sim_n = self.cal_cos_similarity(text,audio,visual,negative_text,negative_audio,negative_visual,negative_label)
        #         # visual = self.v_mlp_layer(visual)
        #         # audio = self.a_mlp_layer(audio)
        #         # text = self.t_mlp_layer(text)
        #         # cos_t_cat = self.t_p_layer(cos_t_cat)
        #         # cos_a_cat = self.a_p_layer(cos_a_cat)
        #         # cos_v_cat = self.v_p_layer(cos_v_cat)
        #         t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
        #         a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
        #         v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
        #     elif negative_label.numel() == 0:
        #         cos_t_cat,cos_a_cat,cos_v_cat ,sim_p= self.cal_cos_similarity(text,audio,visual,positive_text,positive_audio,positive_visual,positive_label)
        #         # visual = self.v_mlp_layer(visual)
        #         # audio = self.a_mlp_layer(audio)
        #         # text = self.t_mlp_layer(text)
        #         # cos_t_cat = self.t_p_layer(cos_t_cat)
        #         # cos_a_cat = self.a_p_layer(cos_a_cat)
        #         # cos_v_cat = self.v_p_layer(cos_v_cat)
        #         t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
        #         a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
        #         v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
        #     else:
        #         cos_t_cat,cos_a_cat,cos_v_cat ,sim_p = self.cal_cos_similarity(text,audio,visual,positive_text,positive_audio,positive_visual,positive_label)
        #         cos_t_cat_n,cos_a_cat_n,cos_v_cat_n ,sim_n = self.cal_cos_similarity(text,audio,visual,negative_text,negative_audio,negative_visual,negative_label)
        #         cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n = self.find_features(cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n,label_ids)
                
                # visual = self.v_mlp_layer(visual)
                # audio = self.a_mlp_layer(audio)
                # text = self.t_mlp_layer(text)
                # cos_t_cat = self.t_p_layer(cos_t_cat)
                # cos_a_cat = self.a_p_layer(cos_a_cat)
                # cos_v_cat = self.v_p_layer(cos_v_cat)
                # cos_t_cat_n = self.t_n_layer(cos_t_cat_n)
                # cos_a_cat_n = self.a_n_layer(cos_a_cat_n)
                # cos_v_cat_n = self.v_n_layer(cos_v_cat_n)
                # t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])+self.cal_l2_loss2(text,cos_t_cat_n[0],cos_t_cat_n[1],cos_t_cat_n[2],0)
                # a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])+self.cal_l2_loss2(audio,cos_a_cat_n[1],cos_a_cat_n[0],cos_a_cat_n[2],1)
                # v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])+self.cal_l2_loss2(visual,cos_v_cat_n[1],cos_v_cat_n[2],cos_v_cat_n[0],2)
            

        # print(cos_t_cat.size())
        # get sample data
        # cos_t_cat = cos_t_cat.permute(1,0,3,2)
        # cos_a_cat = cos_a_cat.permute(1,0,3,2)
        # cos_v_cat = cos_v_cat.permute(1,0,3,2)
        cos_t_cat = torch.stack([text,audio,visual],dim = 0).permute(1,0,3,2)
        cos_a_cat = torch.stack([audio,text,visual],dim = 0).permute(1,0,3,2)
        cos_v_cat = torch.stack([visual,text,audio],dim = 0).permute(1,0,3,2)
        t_s_p, a_s_p, v_s_p = None, None, None
        for idx in range(len(text)):
            x_l_temp, x_a_temp, x_v_temp = self.get_sample_data(
                cos_t_cat[idx],cos_a_cat[idx],cos_v_cat[idx]
            )
            if t_s_p is None:
                t_s_p = x_l_temp
                a_s_p = x_a_temp
                v_s_p = x_v_temp
            else:
                t_s_p = torch.cat([t_s_p, x_l_temp], dim=0)
                a_s_p = torch.cat([a_s_p, x_a_temp], dim=0)
                v_s_p = torch.cat([v_s_p, x_v_temp], dim=0)
        # t_s_p = t_s_p.transpose(1,2)
        # a_s_p = a_s_p.transpose(1,2)
        # v_s_p = v_s_p.transpose(1,2)
        cos_t_cat = cos_t_cat.permute(1,0,3,2)
        cos_a_cat = cos_a_cat.permute(1,0,3,2)
        cos_v_cat = cos_v_cat.permute(1,0,3,2)
        
        
        
        #get modality data
        text = text.transpose(1,2)
        visual = visual.transpose(1, 2)
        audio = audio.transpose(1, 2)
        t_m_p, a_m_p, v_m_p = None, None, None
        for idx in range(len(text)):
            x_l_temp, x_a_temp, x_v_temp = self.get_modality_data(
                text[idx], audio[idx], visual[idx]
            )
            if t_m_p is None:
                t_m_p = x_l_temp
                a_m_p = x_a_temp
                v_m_p = x_v_temp
            else:
                t_m_p = torch.cat([t_m_p, x_l_temp], dim=0)
                a_m_p = torch.cat([a_m_p, x_a_temp], dim=0)
                v_m_p = torch.cat([v_m_p, x_v_temp], dim=0)
        # t_m_p = t_m_p.transpose(1,2)
        # a_m_p = a_m_p.transpose(1,2)
        # v_m_p = v_m_p.transpose(1,2)
        text = text.transpose(1,2)
        visual = visual.transpose(1, 2)
        audio = audio.transpose(1, 2)

        # # print(text.size())
        # # print(t_m_p.size())
        # # print(t_s_p.size())



        #Cross-attention Block
        text = text.permute(1,0,2)
        visual = visual.permute(1,0,2)
        audio = audio.permute(1,0,2)

        t_m_p = t_m_p.permute(1,0,2)
        a_m_p = a_m_p.permute(1,0,2)
        v_m_p = v_m_p.permute(1,0,2)
        # # print(t_m_p.size())

        t_s_p = t_s_p.permute(1,0,2)
        a_s_p = a_s_p.permute(1,0,2)
        v_s_p = v_s_p.permute(1,0,2)

        t_s_output = self.t_s_encoder(t_s_p,text)
        t_m_output = self.t_m_encoder(t_m_p,text)
        a_s_output = self.a_s_encoder(a_s_p,audio)
        a_m_output = self.a_m_encoder(a_m_p,audio)
        v_s_output = self.v_s_encoder(v_s_p,visual)
        v_m_output = self.v_m_encoder(v_m_p,visual)
        # print(t_s_output.size())

        # t_s_output = self.t_s_encoder(t_s_p,text,tgt_key_padding_mask = text_mask)
        # t_m_output = self.t_m_encoder(t_m_p,text,tgt_key_padding_mask = text_mask)
        # a_s_output = self.a_s_encoder(a_s_p,audio,tgt_key_padding_mask = audio_mask)
        # a_m_output = self.a_m_encoder(a_m_p,audio,tgt_key_padding_mask = audio_mask)
        # v_s_output = self.v_s_encoder(v_s_p,visual,tgt_key_padding_mask = visual_mask)
        # v_m_output = self.v_m_encoder(v_m_p,visual,tgt_key_padding_mask = visual_mask)

        t_output = torch.cat([t_m_output, t_s_output], dim=2)
        a_output = torch.cat([a_m_output, a_s_output], dim=2)
        v_output = torch.cat([v_m_output, v_s_output], dim=2)
        # # print(t_output.size())

        t_output = self.t_encoder(t_output)
        a_output = self.a_encoder(a_output)
        v_output = self.v_encoder(v_output)

        # t_output = self.t_encoder(text,src_key_padding_mask = text_mask)
        # a_output = self.a_encoder(audio,src_key_padding_mask = audio_mask)
        # v_output = self.v_encoder(visual,src_key_padding_mask = visual_mask)
        # print(t_output.size())

        # outputs = torch.cat([text_dg[0],visual_dg[0],audio_dg[0]],dim = 1)
        # outputs = torch.stack((t_s_output,t_m_output,a_s_output,a_m_output,v_s_output,v_m_output), dim=1)
        # outputs = t_s_output+t_m_output+a_s_output+a_m_output+v_s_output+v_m_output
        outputs = torch.cat([t_output[0],a_output[0],v_output[0]],dim = 1)
        # outputs = torch.cat((t_s_output,t_m_output,a_s_output,a_m_output,v_s_output,v_m_output), dim=1)
        # outputs = torch.stack((t_s_output,t_m_output,a_s_output,a_m_output), dim=1)
        # outputs = outputs.permute(1,0,2)
        # outputs = self.encoder(outputs)#去掉
        # outputs = outputs[0]
        
        # print(outputs.size())
        logits = self.classifier(outputs)
        # logits = self.classifier(outputs[0])

        # text = text.permute(1, 0 ,2)
        # text_dg = self.tdg_encoder(text, text)

        # visual = visual.permute(1, 0, 2)
        # visual_dg = self.vdg_encoder(visual, visual)
        
        if logits.shape[0] == 1:
            return logits.squeeze().unsqueeze(0)
        else:
            return logits.squeeze()
        # return logits.squeeze()
