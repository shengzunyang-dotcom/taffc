import torch
import torch.nn as nn
from global_configs import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection
from utils.utils import *
# import clip
import torch.nn.functional as F

from pretrained.modeling_bert import BertForSequenceClassification, BertModel
from pretrained.modeling_electra import ElectraForSequenceClassification, ElectraModel, ElectraClassificationHead
from transformers import AutoModel

from src.subnet import *
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

class UnimodalModelText(nn.Module):
    def __init__(self, args):
        super(UnimodalModelText, self).__init__()
        if args.dsbert == 'bert':
            self.bert_encoder = BertModel.from_pretrained(BERT_PRETRAIN_PATH)
        elif args.dsbert == 'electra':
            self.bert_encoder = ElectraModel.from_pretrained(ELECTRA_PRETRAIN_PATH)
        elif args.dsbert == 't5':
            self.bert_encoder = T5EncoderModel.from_pretrained(T5_PRETRAIN_PATH)
        print(args.dsbert)
        self.uni_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.args = args
        self.freeze_params(self.bert_encoder)
    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False
    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids, attention_mask)
        text = text_embeddings[0]
        # print(text.shape)
        # exit(0)
        attention_mask = attention_mask.float()
        text = text.permute(1, 0 ,2)
        text = self.uni_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        fusion = text
        logits = self.classifier(fusion[0])
        outputs = (logits)
        return outputs



class UnimodalModelVisual(nn.Module):
    def __init__(self, args):
        super(UnimodalModelVisual, self).__init__()
        # self.text_encoder = BertModel.from_pretrained(PRETRAIN_PATH)
        # dimension convert
        # domain-specific and domain-general encoder
        self.uni_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.v_dim)
        self.classifier = nn.Linear(args.v_dim, args.ds_label_dim)
        self.args = args

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        visual = visual.permute(1, 0 ,2)
        visual = self.uni_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)
        fusion = visual
        logits = self.classifier(fusion[0])
        outputs = (logits)
        return outputs

class UnimodalModelAudio(nn.Module):
    def __init__(self, args):
        super(UnimodalModelAudio, self).__init__()
        # self.text_encoder = BertModel.from_pretrained(PRETRAIN_PATH)
        # dimension convert
        # domain-specific and domain-general encoder
        self.uni_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.a_dim)
        self.classifier = nn.Linear(args.a_dim, args.ds_label_dim)
        self.args = args

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        audio = audio.permute(1, 0 ,2)
        audio = self.uni_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)
        fusion = audio
        logits = self.classifier(fusion[0])
        outputs = (logits)
        return outputs


class UnimodalModelJoint(nn.Module):
    def __init__(self, args):
        super(UnimodalModelJoint, self).__init__()
        # self.text_encoder = BertModel.from_pretrained(PRETRAIN_PATH)
        # convert layer
        self.v_layer = nn.Linear(args.v_dim, args.t_dim)
        self.a_layer = nn.Linear(args.a_dim, args.t_dim) 
        # dimension convert
        if args.dsbert == 'bert':
            self.bert_encoder = BertModel.from_pretrained(BERT_PRETRAIN_PATH)
        elif args.dsbert == 'electra':
            self.bert_encoder = ElectraModel.from_pretrained(ELECTRA_PRETRAIN_PATH)
        # domain-specific and domain-general encoder
        self.t_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.v_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.a_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)

        self.classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.args = args

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids, attention_mask)
        text = text_embeddings[0]
        visual = self.v_layer(visual)
        audio = self.a_layer(audio)

        attention_mask = attention_mask.float()
        text = text.permute(1, 0 ,2)
        visual = visual.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2)

        text = self.t_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        visual = self.v_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)
        audio = self.a_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)


        fusion = text[0] + visual[0] + audio[0]

        logits = self.classifier(fusion)
        outputs = (logits)
        return outputs



class DGUnimodalText(nn.Module):
    def __init__(self, args):
        super(DGUnimodalText, self).__init__()
        if args.dsbert == 'bert':
            self.bert_encoder = AutoModel.from_pretrained(BERT_PRETRAIN_PATH)
        elif args.dsbert == 'electra':
            self.bert_encoder = AutoModel.from_pretrained(ELECTRA_PRETRAIN_PATH)
        self.tdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.tds_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.csg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        # domian general classfier
        self.args = args
        self.load_dg_model()
        self.tdg_classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.freeze_params(self.tdg_encoder)
        
    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def load_dg_model(self):
        text_save_path = f'./{self.args.checkpoint_dir}/{self.args.step}-text_domain_general.pth'
        text_state_dict = torch.load(text_save_path)
        self.tdg_encoder.load_state_dict(text_state_dict)

        

    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        text_embeddings = self.bert_encoder(input_ids, attention_mask)
        text = text_embeddings[0]
        attention_mask = attention_mask.float()
        text = text.permute(1, 0 ,2)
        text_dg = self.tdg_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        text_ds = self.tds_encoder(text, text, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)
        text_sg = self.csg_encoder(text_dg, text_ds, src_key_padding_mask=attention_mask, tgt_key_padding_mask=attention_mask)

        tdg_logits = self.tdg_classifier(text_sg[0])
        outputs = (tdg_logits)
        return outputs


class DGUnimodalVisual(nn.Module):
    def __init__(self, args):
        super(DGUnimodalVisual, self).__init__()
        self.v_convert_layer = nn.Linear(args.v_dim, args.t_dim)
        self.vdg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.vds_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.csg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.load_dg_model()
        self.args = args
        # domian general classfier
        self.vdg_classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.freeze_params(self.vdg_encoder)
    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False
    def load_dg_model(self):
        visual_save_path = f'./checkpoint-bert/{self.args.step}-visual_domain_general.pth'
        visual_state_dict = torch.load(visual_save_path)
        self.vdg_encoder.load_state_dict(visual_state_dict)
    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        visual = self.v_convert_layer(visual)
        visual_mask = visual_mask.float()
        visual = visual.permute(1, 0, 2)
        
        visual_dg = self.vdg_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)
        visual_ds = self.vds_encoder(visual, visual, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)
        visual_sg = self.csg_encoder(visual_dg, visual_ds, src_key_padding_mask=visual_mask, tgt_key_padding_mask=visual_mask)

        vdg_logits = self.vdg_classifier(visual_sg[0])
        outputs = (vdg_logits)
        return outputs



class DGUnimodalAcoustic(nn.Module):
    def __init__(self, args):
        super(DGUnimodalAcoustic, self).__init__()
        self.a_convert_layer = nn.Linear(args.a_dim, args.t_dim)
        
        self.adg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.ads_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.csg_encoder = nn.Transformer(nhead=8, num_encoder_layers=args.layer, batch_first=False, d_model=args.t_dim)
        self.args = args
        self.adg_classifier = nn.Linear(args.t_dim, args.ds_label_dim)
        self.load_dg_model()
        self.freeze_params(self.adg_encoder)
        
    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def load_dg_model(self):
        audio_save_path = f'./checkpoint-bert/{self.args.step}-audio_domain_general.pth'
        audio_state_dict = torch.load(audio_save_path)
        self.adg_encoder.load_state_dict(audio_state_dict)   
    def forward(self, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None, epoch=0):
        audio = self.a_convert_layer(audio)
        audio_mask = audio_mask.float()
        audio = audio.permute(1, 0, 2)
        audio_dg = self.adg_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)
        audio_ds = self.ads_encoder(audio, audio, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)
        audio_sg = self.csg_encoder(audio_dg, audio_ds, src_key_padding_mask=audio_mask, tgt_key_padding_mask=audio_mask)

        adg_logits = self.adg_classifier(audio_sg[0])

        outputs = (adg_logits)
        return outputs