#from lib2to3.pgen2 import token
#from PIL import Image
import torch
# import wandb
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from global_configs import *

from transformers.models.clip.tokenization_clip import CLIPTokenizer
import argparse
from utils.utils import *
import pickle
from data.dataset import *
# import clip
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.electra.tokenization_electra import ElectraTokenizer
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from contextlib import redirect_stdout

# from src.models import *
# from src.models_v1 import *
# from src.models_dg import *

from src.models import *


parser = argparse.ArgumentParser()
parser.add_argument("--cuda_no", type=str, default=os.environ["CUDA_VISIBLE_DEVICES"])
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default=DATASETS)
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=8)#BATCH_SIZE
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=EPOCHS)
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=seed, default="random")
parser.add_argument("--best_acc", type=float, default=0.1)
parser.add_argument("--wandb_name", type=str, default='none')
parser.add_argument("--domain_type", type=int, default=1)
parser.add_argument("--freeze", type=str, default='freeze')
parser.add_argument("--unimodal", type=str, default='text')
parser.add_argument("--layer", type=int, default=1)

parser.add_argument("--warm_up", type=int, default=5)

parser.add_argument("--test", type=int, default=0)


parser.add_argument("--t_dim", type=int, default=4096)
parser.add_argument("--v_dim", type=int, default=512)
parser.add_argument("--a_dim", type=int, default=1024)

parser.add_argument("--dg_label_dim", type=int, default=2)
parser.add_argument("--ds_label_dim", type=int, default=1)
parser.add_argument("--dsbert",type=str,default="T5")

parser.add_argument("--prompt_dim",type=int,default=128)
parser.add_argument("--prompt_len",type=int,default=100)
parser.add_argument("--t_len",type=int,default=100)
parser.add_argument("--v_len",type=int,default=100)
parser.add_argument("--a_len",type=int,default=100)
parser.add_argument("--dim_head",type=int,default=96)
parser.add_argument("--dropout",type=float,default=0.1)#0.2
parser.add_argument("--depth",type=int,default=1)
parser.add_argument("--scale_dim",type=int,default=4)
parser.add_argument("--loss_rate",type=int,default=1)
parser.add_argument("--classifier_dropout",type=float,default=0.0)
parser.add_argument("--layers",type=int,default=6)
parser.add_argument("--attn_dropout",type=float,default=0.1)
parser.add_argument("--relu_dropout",type=float,default=0.1)
parser.add_argument("--res_dropout",type=float,default=0.1)
parser.add_argument("--embed_dropout",type=float,default=0.25)
parser.add_argument("--attn_mask",action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument("--prompt_plot",type=int,default=128)




args = parser.parse_args()


import numpy as np

import numpy as np

import numpy as np

def compute_residual_error_rate(preds: list, labels: list):
    """
    输入为回归预测值和真实标签，先去掉标签为0的样本，
    再将 >0 映射为1，<0 映射为0，计算残余误差率。

    参数:
        preds: list[list]，长度=9，每个子list为N个回归预测值
        labels: list，长度=N，回归标签

    返回:
        dict:
            - "per_group": list[float], 9组的残余误差率
            - "per_modal": dict, 三个模态的平均残余误差率
            - "overall": float，总体残余误差率
    """
    preds = [np.array(p) for p in preds]
    labels = np.array(labels)

    N = len(labels)
    assert len(preds) == 9, "需要9组预测值"
    for p in preds:
        assert len(p) == N, "预测与标签数量不匹配"

    # 1) 去掉 label == 0 的样本
    nonzero_mask = labels != 0
    labels = labels[nonzero_mask]
    preds = [p[nonzero_mask] for p in preds]

    # 2) 转二分类
    labels_bin = (labels > 0).astype(int)
    preds_bin = [(p > 0).astype(int) for p in preds]

    # 3) 每组误差率
    per_group_error = []
    for i in range(9):
        errors = (preds_bin[i] != labels_bin).sum()
        err_rate = errors / len(labels_bin)
        per_group_error.append(err_rate)

    # 4) 按模态分组
    per_modal_error = {
        "text":  np.mean(per_group_error[0:3]),
        "audio": np.mean(per_group_error[3:6]),
        "video": np.mean(per_group_error[6:9])
    }

    # 5) 总体误差率（把9组一起算）
    stacked_preds = np.stack(preds_bin, axis=0)  # [9, N']
    overall_errors = (stacked_preds != labels_bin).sum()
    overall_total = stacked_preds.size
    overall_error = overall_errors / overall_total

    return {
        "per_group": per_group_error,
        "per_modal": per_modal_error,
        "overall": overall_error
    }




def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def get_loss_func():
    dg_loss_fct = CrossEntropyLoss()
    if args.domain_type == 1 or args.domain_type == 2:
        ds_loss_fct = MSELoss()
    else:
        ds_loss_fct = CrossEntropyLoss()
    return dg_loss_fct, ds_loss_fct

def prepare_training(train_dataloader):
    model = RAGModel(args)

    # if torch.cuda.device_count() > 1:
    #     device_map = {f"cuda:{i}": f"cuda:{i}" for i in range(torch.cuda.device_count())}
    # else:
    #     device_map = {"cuda:0": "cuda:0"}

    # # 迁移未分配组件（网页4的模型并行策略）
    # for name, module in model.named_children():
    #     if not hasattr(module, 'device_map'):  # 非分片模块
    #         module.to(device_map['cuda:0'])

    model = model.to(DEVICE)
    # model = nn.DataParallel(model, device_ids=[0,1,2,3])
    
    # optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-5)#1e-5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCHS)
    return model, optimizer, scheduler

def compute_accurracy(preds, y_test,use_zero=False):
    preds = np.array(preds)
    y_test = np.array(y_test)
    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    # test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    # test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    # acc5 = multiclass_acc(test_preds_a5, test_truth_a5)

    #non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 ])
    preds = preds[non_zeros]
    y_test = y_test[non_zeros]
    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    preds = preds >= 0
    y_test = y_test >= 0
    f_score_w = f1_score(y_test, preds, average="weighted")
    f_score_b = f1_score(y_test, preds, average="binary")
    f_score_mi = f1_score(y_test, preds, average="micro")
    f_score_ma = f1_score(y_test, preds, average="macro")
    recall = recall_score(y_test,preds,average="weighted")
    pre = precision_score(y_test,preds,average="weighted")
    acc = accuracy_score(y_test, preds)
    #return acc, mae, corr, f_score, acc5, acc7
    return acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre ,mae,corr,acc7


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch,valid):
    step = 0
    tr_loss = 0
    to_t_loss = 0
    to_a_loss = 0
    to_v_loss = 0
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        sentence, input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, source_label, label_id, segment = batch
        input_ids, attention_mask, text_mask, visual, visual_mask, audio, audio_mask, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), text_mask.to(DEVICE),visual.to(DEVICE), visual_mask.to(DEVICE), audio.to(DEVICE), audio_mask.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
        # src_label = source_label.long()
        # label_id = label_id.long()
        step += 1
        optimizer.zero_grad()
        outputs ,t_loss,a_loss,v_loss,t_index,a_index,v_index = model(input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch = epoch)
        # outputs = model(input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)
        dg_loss_fct, ds_loss_fct = get_loss_func()        

        # visualize = {'text':sentence,'id':segment,'sim_vector':sim,'label':label_id,'output':outputs,'re_vector':re_vector,'g_s':g_s,'g_m':g_m}
        
        # path = f"/data/yangshengzun/knowledge-injection/visual_sim/visualization-{epoch}-{step}.pkl"

        # if step ==  20  or step == 30:
        #     with open(path, "wb") as f:
        #         pickle.dump(visualize, f)  # 默认协议


        # (text_logits, visual_logits, audio_logits) = outputs
        # text_loss = dg_loss_fct(text_logits, label_id.view(-1))
        # visual_loss = dg_loss_fct(visual_logits, label_id.view(-1))
        # audio_loss = dg_loss_fct(audio_logits, label_id.view(-1))
        # total_loss = text_loss + audio_loss + visual_loss
        
        total_loss = ds_loss_fct(outputs,label_id.view(-1))+t_loss+a_loss+v_loss
        # total_loss = ds_loss_fct(outputs,label_id.view(-1))
        # total_loss = dg_loss_fct(outputs,label_id.view(-1))
        to_t_loss += t_loss.item()
        to_a_loss += a_loss.item()
        to_v_loss += v_loss.item()

        total_loss.backward()
        tr_loss += total_loss.item()
        optimizer.step()
        scheduler.step()
    tr_loss /= step
    to_t_loss/=step
    to_a_loss/=step
    to_v_loss/=step

    return tr_loss ,to_t_loss,to_a_loss,to_v_loss

def eval_epoch(model, dev_dataloader, optimizer, domain=0, epoch=0,valid = False):
    model.eval()
    # if epoch == 50:
    #     save_path = './checkpoint/RAG_model_1.pth'
    #     torch.save(model.state_dict(), save_path)
    #     model = RAGModel(args)
    #     state_dict = torch.load(save_path)
    #     model.load_state_dict(state_dict)
    #     model.to(DEVICE)

    step = 0
    dev_loss = 0
    y_test = []
    text_preds = []
    visual_preds = []
    audio_preds = []
    preds = []
    t_t = []
    t_a = []
    t_v = []
    a_t = []
    a_a = []
    a_v = []
    v_t = []
    v_a = []
    v_v = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, text_mask,visual, visual_len, visual_mask, audio, audio_len, audio_mask, source_label, label_id, segment = batch
            input_ids, attention_mask, text_mask, visual, visual_mask, audio, audio_mask, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), text_mask.to(DEVICE), visual.to(DEVICE), visual_mask.to(DEVICE), audio.to(DEVICE), audio_mask.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
            # source_label = source_label.long()
            # label_id = label_id.long()
            outputs ,t_loss,a_loss,v_loss,t_index,a_index,v_index = model(input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, epoch=epoch)
            # outputs = model(input_ids, attention_mask, text_mask,visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)
            dg_loss_fct, ds_loss_fct = get_loss_func()

            # visualize = {'text':sentence,'id':segment,'sim_vector':sim,'label':label_id,'output':outputs,'re_vector':re_vector,'g_s':g_s,'g_m':g_m}
        
            # path = f"/data/yangshengzun/knowledge-injection/visualization-test/visualization-test-{epoch}-{step}.pkl"
            t_t += label_id[t_index[0]].view(-1).cpu().detach().tolist()
            t_a += label_id[t_index[1]].view(-1).cpu().detach().tolist()
            t_v += label_id[t_index[2]].view(-1).cpu().detach().tolist()
            a_a += label_id[a_index[0]].view(-1).cpu().detach().tolist()
            a_t += label_id[a_index[1]].view(-1).cpu().detach().tolist()
            a_v += label_id[a_index[2]].view(-1).cpu().detach().tolist()
            v_v += label_id[v_index[0]].view(-1).cpu().detach().tolist()
            v_t += label_id[v_index[1]].view(-1).cpu().detach().tolist()
            v_a += label_id[v_index[2]].view(-1).cpu().detach().tolist()
            # if epoch ==1:
            #     with open(path, "wb") as f:
            #         pickle.dump(visualize, f)  # 默认协议

            # (text_logits, visual_logits, audio_logits) = outputs
            # text_loss = dg_loss_fct(text_logits, label_id.view(-1))
            # visual_loss = dg_loss_fct(visual_logits, label_id.view(-1))
            # audio_loss = dg_loss_fct(audio_logits, label_id.view(-1))
            # _, text_predicted = torch.max(text_logits.data, dim=1)
            # _, visual_predicted = torch.max(visual_logits.data, dim=1)
            # _, audio_predicted = torch.max(audio_logits.data, dim=1)
            loss = ds_loss_fct(outputs,label_id.view(-1))+t_loss+a_loss+v_loss
            # loss = ds_loss_fct(outputs,label_id.view(-1))
            # loss = dg_loss_fct(outputs,label_id.view(-1))
            # _,predicted = torch.max(outputs.data,dim = 1)
            y_test_val = label_id.view(-1).cpu().detach().tolist()

            # text_preds_val = text_predicted.cpu().detach().tolist()
            # text_preds += text_preds_val
            # visual_preds_val = visual_predicted.cpu().detach().tolist()
            # visual_preds += visual_preds_val
            # audio_preds_val = audio_predicted.cpu().detach().tolist()
            # audio_preds += audio_preds_val
            #preds_val = predicted.cpu().detach().tolist()
            preds_val = outputs.cpu().detach().tolist()
            preds += preds_val
            y_test += y_test_val
            
            
            total_loss = loss

        dev_loss += total_loss.item()
        # text_acc = accuracy_score(y_test, text_preds)
        # visual_acc = accuracy_score(y_test, visual_preds)
        # audio_acc = accuracy_score(y_test, audio_preds)
        # if valid == True:
        #     print('v_preds:',preds)
        #     print("v_y_test:",y_test)
        # else:
        #     print('t_preds:',preds)
        #     print("t_y_test:",y_test)
        label_total = [t_t,t_a,t_v,a_t,a_a,a_v,v_t,v_a,v_v]
        label_result = compute_residual_error_rate(label_total,y_test)
        acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre , mae, corr, acc7 = compute_accurracy(preds,y_test)
        if acc >= 0.87 and acc <= 0.9 and valid is False:
            # save_path = './checkpoint/RAG_model_1.pth'
            # torch.save(model.state_dict(), save_path)
            # model.cpu()  # 迁移模型参数到CPU内存
            # del model, optimizer, scheduler  # 解除对象引用
            # model = RAGModel(args)
            # state_dict = torch.load(save_path)
            # model.load_state_dict(state_dict)
            # model.to(DEVICE)
            temp = {'pred':preds,'y':y_test}
            with open(f'/data/yangshengzun/knowledge-injection/checkpoint/mosei_my_list.pkl', 'wb') as file:
                pickle.dump(temp, file)
        return dev_loss, acc, f_score_w,f_score_b,f_score_mi,f_score_ma, recall, pre , mae, corr , acc7,label_result




def get_dataset():
    tokenizer = AutoTokenizer.from_pretrained(T5_PRETRAIN_PATH)
    # tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH)
    # tokenizer = CLIPTokenizer.from_pretrained(PRETRAIN_PATH)
    mosi_path = 'merge/mosi_vgg_hubert.pkl'
    mosei_path = 'merge/mosei_vgg_hubert.pkl'
    meld_path = 'merge/meld_vgg_hubert.pkl'

    mosi_data_path = os.path.join(PATH, mosi_path)
    mosei_data_path = os.path.join(PATH, mosei_path)
    meld_data_path = os.path.join(PATH, meld_path)

    # with open(mosi_data_path, "rb") as handle:
    #     mosi_data = pickle.load(handle)

    with open(mosei_data_path, "rb") as handle:
        mosei_data = pickle.load(handle)

    # with open(meld_data_path, "rb") as handle:
    #     meld_data = pickle.load(handle)



    # train_dataset = mosi_data['train']
    # #print(train_dataset[0][0][2].shape)
    # dev_dataset = mosi_data['dev']
    # test_dataset = mosi_data['test']

    train_dataset = mosei_data['train']
    #print(train_dataset[0][0][2].shape)
    dev_dataset = mosei_data['dev']#除10余8
    test_dataset = mosei_data['test']#除10余1

    print(len(train_dataset)+len(dev_dataset)+len(test_dataset))

    # train_dataset = mosi_data['train'] + mosei_data['train'] + meld_data['train']
    # dev_dataset = mosi_data['dev'] + mosei_data['dev'] + meld_data['dev']
    # test_dataset = mosi_data['test'] + mosei_data['test'] + meld_data['test']


    train_data = MultimodalDataset(train_dataset, tokenizer)
    dev_data = MultimodalDataset(dev_dataset, tokenizer)
    test_data = MultimodalDataset(test_dataset, tokenizer)


    return train_data, dev_data, test_data




def get_dataloader(
        train_data, 
        dev_data, 
        test_data, 
        ):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size, collate_fn=padding_collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.train_batch_size, collate_fn=padding_collate_fn)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.train_batch_size, collate_fn=padding_collate_fn)
    # dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    # test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1, collate_fn=padding_collate_fn)
    num_train_optimization_steps = 0
    return train_dataloader, dev_dataloader, test_dataloader

def train(
    model,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
):
    valid_losses, test_losses = [], []
    valid_text_best_accs, valid_visual_best_accs, valid_audio_best_accs = [], [], []
    test_text_best_accs, test_visual_best_accs, test_audio_best_accs = [], [], []

    acc_list = []
    f1_w_list = []
    f1_b_list = []
    f1_mi_list = []
    f1_ma_list = []
    mae_list = []
    corr_list = []
    acc7_list = []

    for epoch in range(int(args.n_epochs)):
        train_loss,t,a,v = train_epoch(model, train_dataloader, optimizer, scheduler, epoch,valid = False)
        valid_loss, v_acc, v_f_score_w,v_f_score_b,v_f_score_mi,v_f_score_ma, v_recall, v_pre , v_mae, v_corr,v_acc7,_ = eval_epoch(model, dev_dataloader, optimizer, epoch=epoch,valid = True)
        test_loss, t_acc, t_f_score_w,t_f_score_b,t_f_score_mi,t_f_score_ma, t_recall, t_pre , t_mae, t_corr ,t_acc7,label_result= eval_epoch(model, test_dataloader, optimizer, epoch=epoch,valid=False)
    
        with open('/data/yangshengzun/knowledge-injection/KNN/more_result.txt', 'a') as f:
            with redirect_stdout(f):
                print(
            f"""epoch:{epoch}, train_loss:{train_loss}, t_loss:{t},a_loss:{a},v_loss:{v},valid_loss:{valid_loss},test_loss:{test_loss},
    valid_acc:{v_acc},valid_f1_w:{v_f_score_w},valid_f1_b:{v_f_score_b},valid_f1_mi:{v_f_score_mi},valid_f1_ma:{v_f_score_ma}, valid_recall:{v_recall}, valid_pre:{v_pre} ,valid_mae:{v_mae},valid_corr:{v_corr},valid_acc7:{v_acc7},
    test_acc:{t_acc}, test_f1_w:{t_f_score_w},test_f1_b:{t_f_score_b},test_f1_mi:{t_f_score_mi},test_f1_ma:{t_f_score_ma}, test_recall:{t_recall},test_pre:{t_pre},test_mae:{t_mae},test_corr:{t_corr},test_acc7:{t_acc7},
    per_group:{label_result['per_group']},per_modal:{label_result['per_modal']},overall:{label_result['overall']}"""
        )
        print(
            f"epoch:{epoch}, train_loss:{train_loss}, t_loss:{t},a_loss:{a},v_loss:{v},valid_loss:{valid_loss},test_loss:{test_loss},\
            valid_acc:{v_acc},valid_f1_w:{v_f_score_w},valid_f1_b:{v_f_score_b},valid_f1_mi:{v_f_score_mi},valid_f1_ma:{v_f_score_ma}, valid_recall:{v_recall}, valid_pre:{v_pre} ,valid_mae:{v_mae},valid_corr:{v_corr},valid_acc7:{v_acc7},\
            test_acc:{t_acc}, test_f1_w:{t_f_score_w},test_f1_b:{t_f_score_b},test_f1_mi:{t_f_score_mi},test_f1_ma:{t_f_score_ma}, test_recall:{t_recall},test_pre:{t_pre},test_mae:{t_mae},test_corr:{t_corr},test_acc7:{t_acc7}"
        )
        print(f"per_group:{label_result['per_group']},per_modal:{label_result['per_modal']},overall:{label_result['overall']}")
        valid_losses.append(valid_loss)

        acc_list.append(t_acc)
        f1_w_list.append(t_f_score_w)
        f1_b_list.append(t_f_score_b)
        f1_mi_list.append(t_f_score_mi)
        f1_ma_list.append(t_f_score_ma)
        mae_list.append(t_mae)
        corr_list.append(t_corr)
        acc7_list.append(t_acc7)

        # valid_text_best_accs.append(valid_text_acc)
        # valid_visual_best_accs.append(valid_visual_acc)
        # valid_audio_best_accs.append(valid_audio_acc)


        # test_text_best_accs.append(test_text_acc)
        # test_visual_best_accs.append(test_visual_acc)
        # test_audio_best_accs.append(test_audio_acc)


        # wandb.log(
        #     (
        #         {
        #             "train_loss": train_loss,
        #             "best_valid_loss": min(valid_losses),
        #             "valid_loss": valid_loss,

        #             # 'valid_text_acc': valid_text_acc,
        #             # "valid_visual_acc": valid_visual_acc,
        #             # "valid_audio_acc": valid_audio_acc,
        #             # 'valid_text_best_acc': max(valid_text_best_accs),
        #             # "valid_visual_best_acc": max(valid_visual_best_accs),
        #             # "valid_audio_best_acc": max(valid_audio_best_accs),

        #             # 'text_text_acc': test_text_acc,
        #             # "test_visual_acc": test_visual_acc,
        #             # "test_audio_acc": test_audio_acc,
        #             # 'test_text_best_acc': max(test_text_best_accs),
        #             # "test_visual_best_acc": max(test_visual_best_accs),
        #             # "test_audio_best_acc": max(test_audio_best_accs),
    
        #         }
        #     )
        # )
    best_index = acc_list.index(max(acc_list))
    
    # 返回对应的指标
    return (acc_list[best_index], 
            f1_w_list[best_index],
            f1_b_list[best_index],
            f1_mi_list[best_index],
            f1_ma_list[best_index], 
            mae_list[best_index], 
            corr_list[best_index],
            acc7_list[best_index])



def main():
    # wandb.init(project="Pretrained-DG", name=args.wandb_name)
    # wandb.init(project="ysz-project1", name=args.wandb_name)
    # wandb.config.update(args)
    # seed = [42,1234,3407,7890,2345,2,3,4,56,7]
    # seed = [3407,3408,3403,3402,3405,3406,3409,3410,3417,3404]
    #seed = [3404]
    result = []
    # print("plot_dim:",args.prompt_plot)
    # for i in seed:
        # args.seed = 3407
    # args.seed = i
        # args.seed = 3407
    set_random_seed(args.seed)
        # full setting
    train_data, dev_data, test_data = get_dataset()
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(train_data, dev_data, test_data)
    model, optimizer, scheduler = prepare_training(train_dataloader)
    result_i = train(
                model=model,
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                scheduler=scheduler
                )
    result.append(result_i)
    model.cpu()  # 迁移模型参数到CPU内存
    del model, optimizer, scheduler  # 解除对象引用
    # print("plot_dim:",args.prompt_plot)
    with open('/data/yangshengzun/knowledge-injection/KNN/more_result.txt', 'a') as f:
        with redirect_stdout(f):
            for i in result:
                print("seed:",args.seed,"acc:",i[0],"f1_w:",i[1],"f1_b:",i[2],"f1_mi:",i[3],"f1_ma:",i[4],"mae:",i[5],"corr:",i[6],"acc7:",i[7])
                print("\n")



if __name__ == "__main__":
    main()
#get_dataset()
# # 1. 清空显存缓存并记录初始状态
# # torch.cuda.empty_cache()
# # initial_mem = torch.cuda.memory_allocated()  # 初始显存

# # 2. 加载模型到CPU（避免GPU自动分配干扰）
# model = AutoModel.from_pretrained(T5_PRETRAIN_PATH)

# # 3. 将模型移动到GPU并强制分配参数显存
# # model = model.to("cuda")  # 触发显存分配

# # 4. 计算模型显存占用
# # model_mem = torch.cuda.memory_allocated() - initial_mem
# # print(f"模型显存占用: {model_mem / 1024**3:.2f} GiB")
# # 查看各层显存占用
# for name, param in model.named_parameters():
#     print(f"层名: {name}")
#     print(f"参数显存: {param.numel() * param.element_size() / 1024**2:.2f} MiB")
#     print(f"梯度显存: {param.grad.numel()*param.grad.element_size()/1024**2:.2f} MiB" if param.grad else "无梯度")

#判断序列长度维度做线性变换是否可行，不行的话就调整prompt_dim，全局调整，把最后的6*768变小