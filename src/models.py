import torch
import torch.nn as nn
from global_configs import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection
from utils.utils import *
# import clip
#import torch.nn.functional as F
from torch import einsum
from einops import rearrange

from pretrained.modeling_bert import BertForSequenceClassification, BertModel
from pretrained.modeling_electra import ElectraForSequenceClassification, ElectraModel, ElectraClassificationHead
from transformers import AutoModel,AutoConfig,T5EncoderModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from src.subnet import *
import torch.nn.functional as F
from modules.transformer import TransformerEncoder,TransformerEncoderLayer
from modules.multihead_attention import MultiheadAttention
import math
from torch.nn import functional as F, init


class KnowledgeInjectionModel(nn.Module):
    def __init__(self, args):
        super(KnowledgeInjectionModel, self).__init__()
        # self.text_encoder = BertModel.from_pretrained(PRETRAIN_PATH)
        if args.dsbert == 'bert':
            self.bert_encoder = BertModel.from_pretrained(BERT_PRETRAIN_PATH)
        elif args.dsbert == 'electra':
            self.bert_encoder = ElectraModel.from_pretrained(ELECTRA_PRETRAIN_PATH)

        self.v_convert_layer = nn.Linear(args.v_dim, args.t_dim)
        self.a_convert_layer = nn.Linear(args.a_dim, args.t_dim)

        # domain-specific and domain-general encoder
        self.tds_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.vds_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.ads_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        self.tdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.vdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.adg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        # stage 1 cross domain
        self.cross_vsg = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.cross_asg = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.cross_tsg = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        # stage 2 language guide
        # self.cross_tv = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        # self.cross_ta = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.cross_tv = nn.MultiheadAttention(args.t_dim, num_heads=8)
        self.cross_ta = nn.MultiheadAttention(args.t_dim, num_heads=8)

        # fusion encoder
        self.fusion_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        # # domain general transformer cls
        # self.vdg_cls = nn.Parameter(torch.rand(1, args.t_dim))
        # self.adg_cls = nn.Parameter(torch.rand(1, args.t_dim))

        # # self.attn = nn.MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, batch_first=False)
        # # domain specific transformer cls
        # self.vds_cls = nn.Parameter(torch.rand(1, args.t_dim))
        # self.ads_cls  = nn.Parameter(torch.rand(1, args.t_dim))

        # domian general classfier
        # self.t_classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        # self.v_classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        # self.a_classifier = nn.Linear(args.t_dim, args.ds_label_dim)

        # domian specific classifier
        # self.ds_classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.classfier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.args = args
        # self.reset_parameters()
        self.load_dg_model()
        self.freeze_params(self.tdg_encoder)
        self.freeze_params(self.vdg_encoder)
        self.freeze_params(self.adg_encoder)

        self.pred_w = nn.Parameter(torch.randn(3))
    
        self.t_ae = VariationalEncoder(args.t_dim, int(args.t_dim/2), int(args.t_dim/4), device=DEVICE)
        self.v_ae = VariationalEncoder(args.t_dim, int(args.t_dim/2), int(args.t_dim/4), device=DEVICE)
        self.a_ae = VariationalEncoder(args.t_dim, int(args.t_dim/2), int(args.t_dim/4), device=DEVICE)

        self.reset_parameter()
        self.kl_loss = nn.KLDivLoss(reduce=False)
        self.ort_loss = DiffLoss()

    def kld(self, feat1, feat2):        
        input = F.log_softmax(feat1, dim=-1)
        target = F.softmax(feat2, dim=-1)
        output = self.kl_loss(input, target)
        output = torch.sum(output, dim=-1)
        return output

    def load_dg_model(self):
        text_save_path = f'./{self.args.checkpoint_dir}/{self.args.step}-text_domain_general.pth'
        text_state_dict = torch.load(text_save_path)
        self.tdg_encoder.load_state_dict(text_state_dict)            
        
        visual_save_path = f'./checkpoint/{self.args.step}-visual_domain_general.pth'
        visual_state_dict = torch.load(visual_save_path)
        self.vdg_encoder.load_state_dict(visual_state_dict)

        audio_save_path = f'./checkpoint/{self.args.step}-audio_domain_general.pth'
        audio_state_dict = torch.load(audio_save_path)
        self.adg_encoder.load_state_dict(audio_state_dict)      

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False


    def reset_parameter(self):
        self.pred_w.data.fill_(1.0)
        # nn.init.uniform_(self.vds_cls)
        # nn.init.uniform_(self.ads_cls)
        # nn.init.uniform_(self.vdg_cls)
        # nn.init.uniform_(self.adg_cls)

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids, attention_mask)
        text = text_embeddings[0]
        visual = self.v_convert_layer(visual)
        audio = self.a_convert_layer(audio)

        attention_mask = attention_mask.float()
        visual_mask = visual_mask.float()
        audio_mask = audio_mask.float()

        # attention_mask = attention_mask.transpose(0 ,1)
        # visual_mask = visual_mask.transpose(0, 1)
        # audio_mask = audio_mask.transpose(0, 1)


        # text encoder, visual encoder, audio encoder
        text = text.permute(1, 0 ,2)
        text_ds = self.tds_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        text_dg = self.tdg_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask) 

        # bsz, seq, dim = visual.shape
        # vds_expanded_param = self.vds_cls.expand(bsz, 1, dim)
        # visual = torch.cat((vds_expanded_param, visual), dim=1)
        visual = visual.permute(1, 0, 2)
        visual_ds = self.vds_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)
        visual_dg = self.vdg_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)

        # bsz, seq, dim = audio.shape
        # ads_expanded_param = self.ads_cls.expand(bsz, 1, dim)
        # audio = torch.cat((ads_expanded_param, audio), dim=1)
        audio = audio.permute(1, 0, 2)
        audio_ds = self.ads_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)
        audio_dg = self.adg_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)

        text_sg = self.cross_tsg(text_dg, text_ds, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        visual_sg = self.cross_vsg(visual_dg, visual_ds, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)
        audio_sg = self.cross_asg(audio_dg, audio_ds, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)

        # print(visual_sg.shape, text_sg.shape, audio_sg.shape, text_sg.shape)
        
        test_diff_loss = self.ort_loss(text_ds, text_dg)
        visual_diff_loss = self.ort_loss(visual_ds, visual_dg)
        audio_diff_loss = self.ort_loss(audio_ds, audio_dg)

        ort_loss = test_diff_loss + visual_diff_loss + audio_diff_loss
        
        tv, tv_attn_weights = self.cross_tv(text_sg, visual_sg, visual_sg, key_padding_mask=visual_mask)
        ta, ta_attn_weights = self.cross_ta(text_sg, audio_sg, audio_sg, key_padding_mask=audio_mask)

        text_sg = text_sg.permute(1, 0 , 2)
        tv = tv.permute(1, 0, 2)
        ta = ta.permute(1, 0, 2)

        zt  = self.t_ae(text_sg)
        ztv = self.v_ae(tv)
        zta = self.a_ae(ta)

        ktv = self.kld(zt, ztv)
        kvt = self.kld(ztv, zt)
        kta = self.kld(zt, zta)
        kat = self.kld(zta, zt)
        
        wtv = (ktv + kvt) / 4
        wta = (kta + kat) / 4
        wt = 1 - (wtv + wta)

        fusion_sg = wtv.unsqueeze(-1) * tv + wta.unsqueeze(-1) * ta + wt.unsqueeze(-1) * text_sg
        fusion_sg = fusion_sg.permute(1, 0, 2)
        fusion = self.fusion_encoder(fusion_sg, fusion_sg, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        # fusion = tv[0] + ta[0] + text_sg[0]
        # t_logits = self.t_classifier(tv[0])
        # v_logits = self.v_classifier(ta[0])
        # a_logits = self.a_classifier(text_sg[0])
        # logits = self.pred_w[0] * t_logits + self.pred_w[1] * v_logits + self.pred_w[2] * a_logits
        logits = self.classfier(fusion[0])
        outputs = (logits, ort_loss)

        return outputs
    
class TextKnowledgeInjectionModel(nn.Module):
    def __init__(self, args):
        super(TextKnowledgeInjectionModel, self).__init__()
        if args.dsbert == 'bert':
            self.bert_encoder = BertModel.from_pretrained(BERT_PRETRAIN_PATH)
        elif args.dsbert == 'electra':
            self.bert_encoder = ElectraModel.from_pretrained(ELECTRA_PRETRAIN_PATH)
        # domain-specific and domain-general encoder
        self.tds_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.tdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.t_classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.args = args
        self.load_dg_model()
        self.freeze_params(self.tdg_encoder)
    def load_dg_model(self):
        text_save_path = f'./checkpoint/{self.args.step}-text_domain_general.pth'
        text_state_dict = torch.load(text_save_path)
        self.tdg_encoder.load_state_dict(text_state_dict)

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids, attention_mask)
        text = text_embeddings[0]
        attention_mask = attention_mask.float()
        text = text.permute(1, 0 ,2)
        text_ds = self.tds_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        text_dg = self.tdg_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        fusion = text_ds + text_dg
        test_diff_loss = self.ort_loss(text_ds, text_dg)
        logits = self.t_classifier(fusion[0])
        outputs = (logits, test_diff_loss)
        return outputs    

    



class UnimodalModel(nn.Module):
    def __init__(self, args):
        super(UnimodalModel, self).__init__()
        if args.dsbert == 'bert':
            self.bert_encoder = BertModel.from_pretrained(BERT_PRETRAIN_PATH)
        elif args.dsbert == 'electra':
            self.bert_encoder = ElectraModel.from_pretrained(ELECTRA_PRETRAIN_PATH)
        # domain-specific and domain-general encoder
        self.tds_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.tdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.t_classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.args = args
        self.load_dg_model()
        self.freeze_params(self.tdg_encoder)
    def load_dg_model(self):
        text_save_path = f'./checkpoint/{self.args.step}-text_domain_general.pth'
        text_state_dict = torch.load(text_save_path)
        self.tdg_encoder.load_state_dict(text_state_dict)
        
        # visual_save_path = f'./checkpoint/{self.args.step}-visual_domain_general.pth'
        # visual_state_dict = torch.load(visual_save_path)
        # self.vdg_encoder.load_state_dict(visual_state_dict)

        # audio_save_path = f'./checkpoint/{self.args.step}-audio_domain_general.pth'
        # audio_state_dict = torch.load(audio_save_path)
        # self.adg_encoder.load_state_dict(audio_state_dict)      

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False


    # def reset_parameter(self):
    #     self.pred_w.data.fill_(1.0)
        # nn.init.uniform_(self.vds_cls)
        # nn.init.uniform_(self.ads_cls)
        # nn.init.uniform_(self.vdg_cls)
        # nn.init.uniform_(self.adg_cls)

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids, attention_mask)
        text = text_embeddings[0]
        attention_mask = attention_mask.float()
        text = text.permute(1, 0 ,2)
        text_ds = self.tds_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        text_dg = self.tdg_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        fusion = text_ds + text_dg

        logits = self.t_classifier(fusion[0])
        outputs = (logits)
        return outputs


class KnowledgeInjectionModelWarmup(nn.Module):
    def __init__(self, args):
        super(KnowledgeInjectionModelWarmup, self).__init__()
        # self.text_encoder = BertModel.from_pretrained(PRETRAIN_PATH)
        
        # dimension convert
        self.bert_encoder = ElectraModel.from_pretrained(PRETRAIN_PATH)
        self.v_convert_layer = nn.Linear(args.v_dim, args.t_dim)
        self.a_convert_layer = nn.Linear(args.a_dim, args.t_dim)

        # domain-specific and domain-general encoder
        self.tds_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.vds_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.ads_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        self.tdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.vdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.adg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        # stage 1 cross domain
        self.cross_vsg = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.cross_asg = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.cross_tsg = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        # stage 2 language guide
        self.cross_tv = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.cross_ta = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        # # domain general transformer cls
        # self.vdg_cls = nn.Parameter(torch.rand(1, args.t_dim))
        # self.adg_cls = nn.Parameter(torch.rand(1, args.t_dim))

        # # self.attn = nn.MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, batch_first=False)
        # # domain specific transformer cls
        # self.vds_cls = nn.Parameter(torch.rand(1, args.t_dim))
        # self.ads_cls  = nn.Parameter(torch.rand(1, args.t_dim))

        # domian general classfier
        self.tdg_classifier = nn.Linear(args.t_dim, args.dg_label_dim)
        self.vdg_classifier = nn.Linear(args.t_dim, args.dg_label_dim)
        self.adg_classifier = nn.Linear(args.t_dim, args.dg_label_dim)

        # domian specific classifier
        self.ds_classifier = nn.Linear(args.t_dim, args.ds_label_dim)

        self.args = args
        # self.reset_parameters()

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def reset_parameters(self):
        nn.init.uniform_(self.vds_cls)
        nn.init.uniform_(self.ads_cls)
        nn.init.uniform_(self.vdg_cls)
        nn.init.uniform_(self.adg_cls)

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids, attention_mask)
        text = text_embeddings[0]
        visual = self.v_convert_layer(visual)
        audio = self.a_convert_layer(audio)

        attention_mask = attention_mask.float()
        visual_mask = visual_mask.float()
        audio_mask = audio_mask.float()

        # attention_mask = attention_mask.transpose(0 ,1)
        # visual_mask = visual_mask.transpose(0, 1)
        # audio_mask = audio_mask.transpose(0, 1)

        if epoch < self.args.warm_up:
            text = text.permute(1, 0 ,2)
            text_dg = self.tdg_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)

            # bsz, seq, dim = visual.shape
            # vdg_expanded_param = self.vdg_cls.expand(bsz, 1, dim)
            # visual = torch.cat((vdg_expanded_param, visual), dim=1)
            visual = visual.permute(1, 0, 2)
            visual_dg = self.vdg_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)

            # bsz, seq, dim = audio.shape
            # adg_expanded_param = self.adg_cls.expand(bsz, 1, dim)
            # audio = torch.cat((adg_expanded_param, audio), dim=1)
            audio = audio.permute(1, 0, 2)
            audio_dg = self.adg_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)


            tdg_logits = self.tdg_classifier(text_dg[0])
            vdg_logits = self.vdg_classifier(visual_dg[0])
            adg_logits = self.adg_classifier(audio_dg[0])

            outputs = (tdg_logits, vdg_logits, adg_logits)
        else:
            # text encoder, visual encoder, audio encoder
            text = text.permute(1, 0 ,2)
            text_ds = self.tds_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
            text_dg = self.tdg_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask) 

            # bsz, seq, dim = visual.shape
            # vds_expanded_param = self.vds_cls.expand(bsz, 1, dim)
            # visual = torch.cat((vds_expanded_param, visual), dim=1)
            visual = visual.permute(1, 0, 2)
            visual_ds = self.vds_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)
            visual_dg = self.vdg_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)

            # bsz, seq, dim = audio.shape
            # ads_expanded_param = self.ads_cls.expand(bsz, 1, dim)
            # audio = torch.cat((ads_expanded_param, audio), dim=1)
            audio = audio.permute(1, 0, 2)
            audio_ds = self.ads_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)
            audio_dg = self.adg_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)

            text_sg = self.cross_tsg(text_dg, text_ds, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
            visual_sg = self.cross_vsg(visual_dg, visual_ds, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)
            audio_sg = self.cross_asg(audio_dg, audio_ds, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)

            # print(visual_sg.shape, text_sg.shape, audio_sg.shape, text_sg.shape)
            
            tv = self.cross_tv(visual_sg, text_sg, src_key_padding_mask=visual_mask, tgt_key_padding_mask=attention_mask)
            ta = self.cross_ta(audio_sg, text_sg, src_key_padding_mask=audio_mask, tgt_key_padding_mask=attention_mask)

            fusion = tv[0] + ta[0] + text_sg[0]
            logits = self.ds_classifier(fusion)
            outputs = (logits)

        return outputs
    

    
class DomainGeneralModel(nn.Module):
    def __init__(self, args):
        super(DomainGeneralModel, self).__init__()
        if args.dsbert == 'bert':
            self.bert_encoder = AutoModel.from_pretrained(BERT_PRETRAIN_PATH)
        elif args.dsbert == 'electra':
            self.bert_encoder = AutoModel.from_pretrained(ELECTRA_PRETRAIN_PATH)

        self.v_convert_layer = nn.Linear(args.v_dim, args.t_dim)
        self.a_convert_layer = nn.Linear(args.a_dim, args.t_dim)

        self.tdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.vdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.adg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        # domian general classfier
        self.tdg_classifier = nn.Linear(args.t_dim, args.dg_label_dim)
        self.vdg_classifier = nn.Linear(args.t_dim, args.dg_label_dim)
        self.adg_classifier = nn.Linear(args.t_dim, args.dg_label_dim)

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False
    

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids, attention_mask)
        text = text_embeddings[0]
        visual = self.v_convert_layer(visual)
        audio = self.a_convert_layer(audio)

        attention_mask = attention_mask.float()
        visual_mask = visual_mask.float()
        audio_mask = audio_mask.float()

        text = text.permute(1, 0 ,2)
        text_dg = self.tdg_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)

        visual = visual.permute(1, 0, 2)
        visual_dg = self.vdg_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)

        audio = audio.permute(1, 0, 2)
        audio_dg = self.adg_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)

        tdg_logits = self.tdg_classifier(text_dg[0])
        vdg_logits = self.vdg_classifier(visual_dg[0])
        adg_logits = self.adg_classifier(audio_dg[0])

        outputs = (tdg_logits, vdg_logits, adg_logits)
        return outputs

class RAGModel(nn.Module):
    def __init__(self, args):
        super(RAGModel, self).__init__()
        if args.dsbert == 'bert':
            self.bert_encoder = AutoModel.from_pretrained(BERT_PRETRAIN_PATH)
        elif args.dsbert == 'electra':
            self.bert_encoder = AutoModel.from_pretrained(ELECTRA_PRETRAIN_PATH)
        elif args.dsbert == 'T5':
            # model_config = AutoConfig.from_pretrained(T5_PRETRAIN_PATH)
            # with init_empty_weights():
            self.bert_encoder = T5EncoderModel.from_pretrained(T5_PRETRAIN_PATH)
            # self.bert_encoder = AutoModel.from_pretrained(T5_PRETRAIN_PATH)
            # self.bert_encoder = load_checkpoint_and_dispatch(
            # self.bert_encoder, 
            # checkpoint=T5_PRETRAIN_PATH, 
            # device_map="balanced", 
            # no_split_module_classes=['T5Block'],
            #     )
            self.freeze_params(self.bert_encoder)
            
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

        # self.a_lvp = MLPLayer(
        #     args.prompt_len + args.a_len + args.t_len + args.v_len, args.a_len, True
        # )
        # self.v_alp = MLPLayer(
        #     args.prompt_len + args.a_len + args.t_len + args.v_len, args.v_len, True
        # )
        # self.l_avp = MLPLayer(
        #     args.prompt_len + args.a_len + args.t_len + args.v_len, args.t_len, True
        # )
        self.a_lvp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.v_alp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )
        self.l_avp = MLPLayer(
            args.prompt_dim * 4, args.prompt_dim  , True
        )



        # self.t_tavp = MLPLayer(
        #     args.prompt_len + args.t_len + args.a_len + args.v_len, args.t_len, True
        # )
        # self.a_tavp = MLPLayer(
        #     args.prompt_len + args.t_len + args.a_len + args.v_len, args.a_len, True
        # )
        # self.v_tavp = MLPLayer(
        #     args.prompt_len + args.a_len + args.a_len + args.v_len, args.v_len, True
        # )

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
        # self.sample_prompt = nn.Parameter(torch.randn((3,args.prompt_dim,args.prompt_len)))#可训练参数初始化
        # self.t_s_encoder  = TransformerEncoder(
        #                     embed_dim=args.prompt_dim,
        #                     num_heads=8,
        #                     layers = args.depth,
        #                     attn_dropout=args.attn_dropout,
        #                     relu_dropout = args.relu_dropout,
        #                     res_dropout = args.res_dropout,
        #                     atten_mask=False
        #                     )
        self.t_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.t_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.a_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.a_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.v_s_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        self.v_m_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, num_decoder_layers = args.layers, batch_first=False, d_model=args.prompt_dim)
        # self.classifier = nn.Linear(args.prompt_dim,args.ds_label_dim)
        # self.encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.prompt_dim)
        t_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.t_encoder = nn.TransformerEncoder(
            encoder_layer=t_encoder_layer,
            num_layers=args.layers
        )
        a_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.a_encoder = nn.TransformerEncoder(
            encoder_layer=a_encoder_layer,
            num_layers=args.layers
        )
        v_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.prompt_dim,
            nhead=8,
            dim_feedforward=args.prompt_dim*4,
            dropout=args.dropout
        )
        self.v_encoder = nn.TransformerEncoder(
            encoder_layer=v_encoder_layer,
            num_layers=args.layers
        )

        self.classifier = T5ClassificationHead(args)

        self.t_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        self.a_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        self.v_mlp_layer = nn.Linear(args.prompt_dim,args.prompt_dim)

        self.t_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        self.a_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        self.v_p_layer = nn.Linear(args.prompt_dim,args.prompt_dim)

        self.t_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        self.a_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)
        self.v_n_layer = nn.Linear(args.prompt_dim,args.prompt_dim)

    



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
        
        t_index = torch.stack((most_similar_1_temp, most_similar_4, most_similar_5), dim=0)
        a_index = torch.stack((most_similar_2_temp, most_similar_6, most_similar_7), dim=0)
        v_index = torch.stack((most_similar_3_temp, most_similar_8, most_similar_9), dim=0)
        sim = torch.stack([cosine_sim_1,cosine_sim_4,cosine_sim_5,cosine_sim_2,cosine_sim_6,cosine_sim_7,cosine_sim_3,cosine_sim_8,cosine_sim_9],dim = 0)

        return cos_t_cat,cos_a_cat,cos_v_cat,sim,t_index,a_index,v_index
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
    
    def cal_l2_loss1(self,x1,x2,x3,x4):
        loss1 = torch.norm((x1-x2),p=2)
        loss2 = torch.norm((x1-x3),p=2)
        loss3 = torch.norm((x1-x4),p=2)
        # print(torch.norm((x1-x2),p=2))
        # print(torch.norm((x1-x3),p=2))
        # print(torch.norm((x1-x4),p=2))
        return 0.001*(loss1 + loss2 + loss3)
    
    def cal_l2_loss2(self,x1,x2,x3,x4,modality_type):
        if modality_type == 0:
            #200,300,200
            loss1 = max(0,450-torch.norm((x1-x2),p=2))#450
            loss2 = max(0,500-torch.norm((x1-x3),p=2))#500
            loss3 = max(0,450-torch.norm((x1-x4),p=2))#450
        elif modality_type == 1:
            #300,400,300
            loss1 = max(0,300-torch.norm((x1-x2),p=2))#300
            loss2 = max(0,500-torch.norm((x1-x3),p=2))#500
            loss3 = max(0,300-torch.norm((x1-x4),p=2))#300
        else:
            #200,300,200
            loss1 = max(0,150-torch.norm((x1-x2),p=2))#150
            loss2 = max(0,500-torch.norm((x1-x3),p=2))#500
            loss3 = max(0,300-torch.norm((x1-x4),p=2))#300
        
        # print(torch.norm((x1-x2),p=2))
        # print(torch.norm((x1-x3),p=2))
        # print(torch.norm((x1-x4),p=2))
        return 0.001*(loss1 + loss2 + loss3)
    
    def find_features(self,cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n,labels):
        t_cat,a_cat,v_cat = None, None, None
        t_cat_n,a_cat_n,v_cat_n = None, None, None
        cos_t_cat = torch.transpose(cos_t_cat, 0, 1)
        cos_a_cat = torch.transpose(cos_a_cat, 0, 1)
        cos_v_cat = torch.transpose(cos_v_cat, 0, 1)

        cos_t_cat_n = torch.transpose(cos_t_cat_n, 0, 1)
        cos_a_cat_n = torch.transpose(cos_a_cat_n, 0, 1)
        cos_v_cat_n = torch.transpose(cos_v_cat_n, 0, 1)
        for i in range(len(labels)):
            if labels[i] >= 0:
                if t_cat is None:
                    t_cat = cos_t_cat[i].unsqueeze(0)
                    a_cat = cos_a_cat[i].unsqueeze(0)
                    v_cat = cos_v_cat[i].unsqueeze(0)
                    t_cat_n = cos_t_cat_n[i].unsqueeze(0)
                    a_cat_n = cos_a_cat_n[i].unsqueeze(0)
                    v_cat_n = cos_v_cat_n[i].unsqueeze(0)
                else:
                    t_cat = torch.cat([t_cat, cos_t_cat[i].unsqueeze(0)], dim=0)
                    a_cat = torch.cat([a_cat, cos_a_cat[i].unsqueeze(0)], dim=0)
                    v_cat = torch.cat([v_cat, cos_v_cat[i].unsqueeze(0)], dim=0)
                    t_cat_n = torch.cat([t_cat_n, cos_t_cat_n[i].unsqueeze(0)], dim=0)
                    a_cat_n = torch.cat([a_cat_n, cos_a_cat_n[i].unsqueeze(0)], dim=0)
                    v_cat_n = torch.cat([v_cat_n, cos_v_cat_n[i].unsqueeze(0)], dim=0)
            else:
                if t_cat is None:
                    t_cat = cos_t_cat_n[i].unsqueeze(0)
                    a_cat = cos_a_cat_n[i].unsqueeze(0)
                    v_cat = cos_v_cat_n[i].unsqueeze(0)
                    t_cat_n = cos_t_cat[i].unsqueeze(0)
                    a_cat_n = cos_a_cat[i].unsqueeze(0)
                    v_cat_n = cos_v_cat[i].unsqueeze(0)
                else:
                    t_cat = torch.cat([t_cat, cos_t_cat_n[i].unsqueeze(0)], dim=0)
                    a_cat = torch.cat([a_cat, cos_a_cat_n[i].unsqueeze(0)], dim=0)
                    v_cat = torch.cat([v_cat, cos_v_cat_n[i].unsqueeze(0)], dim=0)
                    t_cat_n = torch.cat([t_cat_n, cos_t_cat[i].unsqueeze(0)], dim=0)
                    a_cat_n = torch.cat([a_cat_n, cos_a_cat[i].unsqueeze(0)], dim=0)
                    v_cat_n = torch.cat([v_cat_n, cos_v_cat[i].unsqueeze(0)], dim=0)
        t_cat = torch.transpose(t_cat, 0, 1)
        a_cat = torch.transpose(a_cat, 0, 1)
        v_cat = torch.transpose(v_cat, 0, 1)

        t_cat_n = torch.transpose(t_cat_n, 0, 1)
        a_cat_n = torch.transpose(a_cat_n, 0, 1)
        v_cat_n = torch.transpose(v_cat_n, 0, 1)
        return t_cat,a_cat,v_cat,t_cat_n,a_cat_n,v_cat_n
    
    def random_re(self,x_t_re,x_a_re,x_v_re,k):
        assert x_t_re.shape == x_a_re.shape == x_v_re.shape, "输入张量维度必须一致"
        b, n, d = x_t_re.shape
        
        results = []
        
        # 生成随机索引（允许重复采样）[7](@ref)
        indices1 = torch.randint(0, b, (k,))
        indices2 = torch.randint(0, b, (k,))
        indices3 = torch.randint(0, b, (k,))
        indices4 = torch.randint(0, b, (k,))
        indices5 = torch.randint(0, b, (k,))
        indices6 = torch.randint(0, b, (k,))
        indices7 = torch.randint(0, b, (k,))
        indices8 = torch.randint(0, b, (k,))
        indices9 = torch.randint(0, b, (k,))
        
        # 多模态同步采样 [2,6](@ref)
        sampled_t = x_t_re[indices1]  # [k, n, d]
        sampled_a = x_a_re[indices2]
        sampled_v = x_v_re[indices3]
        t = torch.stack([sampled_t, sampled_a, sampled_v], dim=0)  # [3, k, n, d]
        # 多模态同步采样 [2,6](@ref)
        sampled_t = x_t_re[indices4]  # [k, n, d]
        sampled_a = x_a_re[indices5]
        sampled_v = x_v_re[indices6]
        a = torch.stack([sampled_a, sampled_t, sampled_v], dim=0)  # [3, k, n, d]
        # 多模态同步采样 [2,6](@ref)
        sampled_t = x_t_re[indices7]  # [k, n, d]
        sampled_a = x_a_re[indices8]
        sampled_v = x_v_re[indices9]
        v = torch.stack([sampled_v,sampled_t, sampled_a ], dim=0)  # [3, k, n, d]
        
        return t,a,v


    def forward(self, input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids = input_ids, attention_mask = attention_mask)
        text = text_embeddings[0]
        visual = self.v_convert_layer(visual)
        audio = self.a_convert_layer(audio)
        text = self.t_convert_layer(text)
        
        # attention_mask = attention_mask.float()
        # visual_mask = visual_mask.float()
        # audio_mask = audio_mask.float()

        sim_p = None
        sim_n = None
        t_index,a_index,v_index = None,None,None
        
        if label_ids is None :
            cos_t_cat,cos_a_cat,cos_v_cat ,sim_p,t_index,a_index,v_index = self.cal_cos_similarity(text,audio,visual,text,audio,visual)
            # visual = self.v_mlp_layer(visual)
            # audio = self.a_mlp_layer(audio)
            # text = self.t_mlp_layer(text)
            cos_t_cat = self.t_p_layer(cos_t_cat)
            cos_a_cat = self.a_p_layer(cos_a_cat)
            cos_v_cat = self.v_p_layer(cos_v_cat)
            t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
            a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
            v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
        else :
            label_ids = label_ids.squeeze()
            if len(label_ids.shape) == 0 :
                label_ids = label_ids.unsqueeze(0)
            positive_label = torch.nonzero(label_ids >= 0).squeeze()
            negative_label = torch.nonzero(label_ids < 0).squeeze()
            if len(positive_label.shape) == 0 :
                positive_label = positive_label.unsqueeze(0)
            if len(negative_label.shape) == 0 :
                negative_label = negative_label.unsqueeze(0)
            positive_text,positive_visual,positive_audio = text[positive_label],visual[positive_label],audio[positive_label]
            negative_text,negative_visual,negative_audio = text[negative_label],visual[negative_label],audio[negative_label]
            if positive_label.numel() == 0:
                cos_t_cat,cos_a_cat,cos_v_cat,sim_n,t_index,a_index,v_index = self.cal_cos_similarity(text,audio,visual,negative_text,negative_audio,negative_visual,negative_label)
                # visual = self.v_mlp_layer(visual)
                # audio = self.a_mlp_layer(audio)
                # text = self.t_mlp_layer(text)
                cos_t_cat = self.t_p_layer(cos_t_cat)
                cos_a_cat = self.a_p_layer(cos_a_cat)
                cos_v_cat = self.v_p_layer(cos_v_cat)
                t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
                a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
                v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
            elif negative_label.numel() == 0:
                cos_t_cat,cos_a_cat,cos_v_cat ,sim_p,t_index,a_index,v_index= self.cal_cos_similarity(text,audio,visual,positive_text,positive_audio,positive_visual,positive_label)
                # visual = self.v_mlp_layer(visual)
                # audio = self.a_mlp_layer(audio)
                # text = self.t_mlp_layer(text)
                # cos_t_cat = self.t_p_layer(cos_t_cat)
                # cos_a_cat = self.a_p_layer(cos_a_cat)
                # cos_v_cat = self.v_p_layer(cos_v_cat)
                cos_t_cat = self.t_n_layer(cos_t_cat)
                cos_a_cat = self.a_n_layer(cos_a_cat)
                cos_v_cat = self.v_n_layer(cos_v_cat)
                t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])
                a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])
                v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])
            else:
                cos_t_cat,cos_a_cat,cos_v_cat ,sim_p,t_index,a_index,v_index = self.cal_cos_similarity(text,audio,visual,positive_text,positive_audio,positive_visual,positive_label)
                cos_t_cat_n,cos_a_cat_n,cos_v_cat_n ,sim_n ,t_index,a_index,v_index= self.cal_cos_similarity(text,audio,visual,negative_text,negative_audio,negative_visual,negative_label)
                cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n = self.find_features(cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n,label_ids)
                
                # visual = self.v_mlp_layer(visual)
                # audio = self.a_mlp_layer(audio)
                # text = self.t_mlp_layer(text)
                cos_t_cat = self.t_p_layer(cos_t_cat)
                cos_a_cat = self.a_p_layer(cos_a_cat)
                cos_v_cat = self.v_p_layer(cos_v_cat)
                cos_t_cat_n = self.t_n_layer(cos_t_cat_n)
                cos_a_cat_n = self.a_n_layer(cos_a_cat_n)
                cos_v_cat_n = self.v_n_layer(cos_v_cat_n)
                t_loss = self.cal_l2_loss1(text,cos_t_cat[0],cos_t_cat[1],cos_t_cat[2])+self.cal_l2_loss2(text,cos_t_cat_n[0],cos_t_cat_n[1],cos_t_cat_n[2],0)
                a_loss = self.cal_l2_loss1(audio,cos_a_cat[1],cos_a_cat[0],cos_a_cat[2])+self.cal_l2_loss2(audio,cos_a_cat_n[1],cos_a_cat_n[0],cos_a_cat_n[2],1)
                v_loss = self.cal_l2_loss1(visual,cos_v_cat[1],cos_v_cat[2],cos_v_cat[0])+self.cal_l2_loss2(visual,cos_v_cat_n[1],cos_v_cat_n[2],cos_v_cat_n[0],2)
            
        # if label_ids is None :
        #     cos_t_cat,cos_a_cat,cos_v_cat = self.random_re(text,audio,visual,text.shape[0])
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
        #         cos_t_cat,cos_a_cat,cos_v_cat = self.random_re(negative_text,negative_audio,negative_visual,negative_text.shape[0])
        #     elif negative_label.numel() == 0:
        #         cos_t_cat,cos_a_cat,cos_v_cat = self.random_re(positive_text,positive_audio,positive_visual,positive_text.shape[0])
        #     else:
        #         k = text.shape[0]
        #         cos_t_cat,cos_a_cat,cos_v_cat = self.random_re(positive_text,positive_audio,positive_visual,k)
        #         cos_t_cat_n,cos_a_cat_n,cos_v_cat_n = self.random_re(negative_text,negative_audio,negative_visual,k)
        #         cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n = self.find_features(cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n,label_ids)



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

        t_s_output = self.t_s_encoder(t_s_p,text,src_key_padding_mask = text_mask,tgt_key_padding_mask = text_mask)
        t_m_output = self.t_m_encoder(t_m_p,text,src_key_padding_mask = text_mask,tgt_key_padding_mask = text_mask)
        a_s_output = self.a_s_encoder(a_s_p,audio,src_key_padding_mask = audio_mask,tgt_key_padding_mask = audio_mask)
        a_m_output = self.a_m_encoder(a_m_p,audio,src_key_padding_mask = audio_mask,tgt_key_padding_mask = audio_mask)
        v_s_output = self.v_s_encoder(v_s_p,visual,src_key_padding_mask = visual_mask,tgt_key_padding_mask = visual_mask)
        v_m_output = self.v_m_encoder(v_m_p,visual,src_key_padding_mask = visual_mask,tgt_key_padding_mask = visual_mask)
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

        # t_output = self.t_encoder(t_output,src_key_padding_mask = text_mask)
        # a_output = self.a_encoder(a_output,src_key_padding_mask = audio_mask)
        # v_output = self.v_encoder(v_output,src_key_padding_mask = visual_mask)

        t_output = self.t_encoder(text,src_key_padding_mask = text_mask)
        a_output = self.a_encoder(audio,src_key_padding_mask = audio_mask)
        v_output = self.v_encoder(visual,src_key_padding_mask = visual_mask)
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

        # if sim_n is None and sim_p is None:
        #     sim = sim
        #     re_vector = torch.stack([cos_t_cat,cos_a_cat,cos_v_cat],dim = 0)
        # if sim_n is None:
        #     sim = sim_p
        #     re_vector = torch.stack([cos_t_cat,cos_a_cat,cos_v_cat],dim = 0)
        # elif sim_p is None:
        #     sim = sim_n
        #     re_vector = torch.stack([cos_t_cat,cos_a_cat,cos_v_cat],dim = 0)
        # else :
        #     sim = [sim_p,sim_n]
        #     re_vector = torch.stack([cos_t_cat,cos_a_cat,cos_v_cat,cos_t_cat_n,cos_a_cat_n,cos_v_cat_n],dim = 0)
        
        # g_s = torch.stack([t_s_p,a_s_p,v_s_p],dim = 0)
        # g_m = torch.stack([t_m_p,a_m_p,v_m_p],dim = 0)
        if logits.shape[0] == 1:
            return logits.squeeze().unsqueeze(0) ,t_loss,a_loss,v_loss,t_index,a_index,v_index
        else:
            return logits.squeeze(),t_loss,a_loss,v_loss,t_index,a_index,v_index
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

# class CrossBlock(nn.Module):
#     def __init__(self, dim, heads = 8, dropout = 0.):
#         super().__init__()
#         self.atten = nn.MultiheadAttention(embed_dim = dim, num_heads = heads, dropout=dropout)
#         self.ln = nn.LayerNorm(dim)
#         self.ff = FeedForward(dim, dim*4, dropout = 0.)
#     def forward(self, x_q,x_k,x_v):
#         residual = x_q
#         x_q = self.ln(x_q)
#         x_k = self.ln(x_k)
#         x_v = self.ln(x_v)
#         out,_ = self.atten(x_q,x_k,x_v)
#         out = out + residual
#         out = self.ff(out)
#         return out

# class TransformerEncoder(nn.Module):
#     def __init__(self,dim, num_heads, p_dropout, depth, hidden_dim=None):
#         super().__init__()
#         self.encoder = nn.ModuleList()
#         for _ in range(depth):
#             self.encoder.append(
#                 CrossBlock(
#                     dim, num_heads,  p_dropout
#                 )
#             )

#     def forward(self, x_q,x_k,x_v):
#         for blk in self.encoder:
#             x_q = blk(x_q,x_k,x_v)
#         return x_q

class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.prompt_dim*3 ,args.prompt_dim*6)
        self.dropout = nn.Dropout(p=args.classifier_dropout)
        self.out_proj = nn.Linear(args.prompt_dim*6, args.ds_label_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         # self.norm2 = nn.LayerNorm(dim)
#         # self.norm3 = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x_q,x_k,x_v, **kwargs):
#         return self.fn(self.norm(x_q), self.norm(x_k),self.norm(x_v),**kwargs)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_q = nn.Linear(dim, inner_dim, bias = False)
#         self.to_k = nn.Linear(dim, inner_dim, bias = False)
#         self.to_v = nn.Linear(dim, inner_dim, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x_q,x_k,x_v):
#         h = self.heads
#         # qkv = self.to_qkv(x).chunk(3, dim = -1)
#         x_q = self.to_q(x_q)
#         x_k = self.to_k(x_k)
#         x_v = self.to_v(x_v)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [x_q,x_k,x_v])
#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#         attn = dots.softmax(dim=-1)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out =  self.to_out(out)
#         return out

# class IntraEncoderBlock(nn.Module):
#     def __init__(
#         self, dim, num_heads, head_dim, p_dropout,  hidden_dim=None
#     ):
#         super(IntraEncoderBlock, self).__init__()

#         self.self_attention = PreNorm(
#             dim=dim, fn=Attention(dim, num_heads, head_dim, p_dropout)
#         )
#         self.intra_attention = PreNorm(
#             dim=dim, fn=Attention(dim, num_heads, head_dim, p_dropout)
#         )

#         self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim)

#     def forward(self, x_q,x_k,x_v):
        
#         x_q = self.intra_attention(x_q,x_k,x_v) + x_q   # b*t s d


#         x = self.mlp(x_q) + x_q
        
#         return x

# class CrossEncoderBlock(nn.Module):
#     def __init__(
#         self, dim, num_heads, head_dim, p_dropout,  hidden_dim=None
#     ):
#         super(CrossEncoderBlock, self).__init__()

#         self.cross_attention = PreNorm(
#             dim=dim, fn=Attention(dim, num_heads, head_dim, p_dropout)
#         )
#         self.self_attention = PreNorm(
#             dim=dim, fn=Attention(dim, num_heads, head_dim, p_dropout)
#         )

#         self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim)

#     def forward(self, x_q,x_k,x_v):

#         # b, n, s, d = x.shape    # b t s d 
#         # x = torch.flatten(x, start_dim=0, end_dim=1)  # 1×nt·nh·nw·d --> nt×nh·nw·d     b*t s d

#         # x = self.intra_attention(x) + x   # b*t s d
#         # x = x.transpose(1,2)
#         # x = x.reshape(b, n, s, d).transpose(1, 2)   # b s t d
#         # x = torch.flatten(x, start_dim=0, end_dim=1)  # nt×nh·nw·d --> nh·nw×nt·d       b*s t d

#         x_q = self.cross_attention(x_q,x_k,x_v) + x_q  # b*s t d

#         x = self.mlp(x_q) + x_q
#         # x = x.reshape(b, n, s, d)
#         # x = x.transpose(1,2)  # b n s d  # reshaping because this block is used for several depths in CrossIntraEncoder class and Next layer will expect the x in proper shape

#         return x


# class IntraEncoder(nn.Module):
#     def __init__(
#         self, dim, num_heads, head_dim, p_dropout, depth,  hidden_dim=None
#     ):
#         super(IntraEncoder, self).__init__()
#         self.encoder = nn.ModuleList()

    #     for _ in range(depth):
    #         self.encoder.append(
    #             IntraEncoderBlock(
    #                 dim, num_heads, head_dim, p_dropout, hidden_dim
    #             )
    #         )

    # def forward(self, x_q,x_k,x_v):

    #     # b = x_q.shape[0]
    #     for blk in self.encoder:
    #         x_q = blk(x_q,x_k,x_v)
    #     return x_q

# class CrossEncoder(nn.Module):
#     def __init__(
#         self, dim, num_heads, head_dim, p_dropout, depth,  hidden_dim=None
#     ):
#         super(CrossEncoder, self).__init__()
#         self.encoder = nn.ModuleList()

#         for _ in range(depth):
#             self.encoder.append(
#                 CrossEncoderBlock(
#                     dim, num_heads, head_dim, p_dropout, hidden_dim
#                 )
#             )

#     def forward(self, x_q,x_k,x_v):

#         # b = x.shape[0]
#         for blk in self.encoder:
#             x_q = blk(x_q,x_k,x_v)
#         return x_q
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
#         dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
#         attn = dots.softmax(dim=-1)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out =  self.to_out(out)
#         return out

# class IntraEncoderBlock(nn.Module):
#     def __init__(
#         self, dim, num_heads, head_dim, p_dropout,  hidden_dim=None
#     ):
#         super(IntraEncoderBlock, self).__init__()

#         self.cross_attention = PreNorm(
#             dim=dim, fn=Attention(dim, num_heads, head_dim, p_dropout)
#         )
#         self.intra_attention = PreNorm(
#             dim=dim, fn=Attention(dim, num_heads, head_dim, p_dropout)
#         )

#         self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim)

#     def forward(self, x):

#         b, n, s, d = x.shape    # b t s d 
#         x = x.transpose(1,2)
#         x = torch.flatten(x, start_dim=0, end_dim=1)  # 1×nt·nh·nw·d --> nt×nh·nw·d     b*t s d
        
#         x = self.intra_attention(x) + x   # b*t s d

#         # x = x.reshape(b, n, s, d).transpose(1, 2)   # b s t d
#         # x = torch.flatten(x, start_dim=0, end_dim=1)  # nt×nh·nw·d --> nh·nw×nt·d       b*s t d


#         # x = self.cross_attention(x) + x  # b*s t d

#         x = self.mlp(x) + x
#         x = x.reshape(b, s, n, d)
#         x = x.transpose(1,2)  # b n s d  # reshaping because this block is used for several depths in CrossIntraEncoder class and Next layer will expect the x in proper shape

#         return x

# class CrossEncoderBlock(nn.Module):
#     def __init__(
#         self, dim, num_heads, head_dim, p_dropout,  hidden_dim=None
#     ):
#         super(CrossEncoderBlock, self).__init__()

#         self.cross_attention = PreNorm(
#             dim=dim, fn=Attention(dim, num_heads, head_dim, p_dropout)
#         )
#         self.intra_attention = PreNorm(
#             dim=dim, fn=Attention(dim, num_heads, head_dim, p_dropout)
#         )

#         self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim)

#     def forward(self, x):

#         b, n, s, d = x.shape    # b t s d 
#         # x = torch.flatten(x, start_dim=0, end_dim=1)  # 1×nt·nh·nw·d --> nt×nh·nw·d     b*t s d

#         # x = self.intra_attention(x) + x   # b*t s d
#         x = x.transpose(1,2)
#         # x = x.reshape(b, n, s, d).transpose(1, 2)   # b s t d
#         x = torch.flatten(x, start_dim=0, end_dim=1)  # nt×nh·nw·d --> nh·nw×nt·d       b*s t d


#         x = self.cross_attention(x) + x  # b*s t d

#         x = self.mlp(x) + x
#         x = x.reshape(b, s, n, d)
#         x = x.transpose(1,2)  # b n s d  # reshaping because this block is used for several depths in CrossIntraEncoder class and Next layer will expect the x in proper shape

#         return x


# class IntraEncoder(nn.Module):
#     def __init__(
#         self, dim, num_heads, head_dim, p_dropout, depth,  hidden_dim=None
#     ):
#         super(IntraEncoder, self).__init__()
#         self.encoder = nn.ModuleList()

#         for _ in range(depth):
#             self.encoder.append(
#                 IntraEncoderBlock(
#                     dim, num_heads, head_dim, p_dropout, hidden_dim
#                 )
#             )

#     def forward(self, x):

#         b = x.shape[0]
#         for blk in self.encoder:
#             x = blk(x)
#         return x

# class CrossEncoder(nn.Module):
#     def __init__(
#         self, dim, num_heads, head_dim, p_dropout, depth,  hidden_dim=None
#     ):
#         super(CrossEncoder, self).__init__()
#         self.encoder = nn.ModuleList()

#         for _ in range(depth):
#             self.encoder.append(
#                 CrossEncoderBlock(
#                     dim, num_heads, head_dim, p_dropout, hidden_dim
#                 )
#             )

#     def forward(self, x):

#         b = x.shape[0]
#         for blk in self.encoder:
#             x = blk(x)
#         return x


