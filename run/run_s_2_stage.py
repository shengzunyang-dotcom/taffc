from PIL import Image
import torch
import wandb
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from global_configs import *
# from src.models import *
from transformers.models.clip.tokenization_clip import CLIPTokenizer
import argparse
from utils.utils import *
import pickle
from data.dataset import *
# import clip
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from sklearn.metrics import accuracy_score, f1_score
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.electra.tokenization_electra import ElectraTokenizer
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
# from src.model_update import *
# from src.model_dg import *
# from src.model_v1 import *
from src.models_2_stage import *
parser = argparse.ArgumentParser()
parser.add_argument("--cuda_no", type=str, default=os.environ["CUDA_VISIBLE_DEVICES"])
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default=DATASETS)
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=BATCH_SIZE)
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
parser.add_argument("--warm_up", type=int, default=0)


args = parser.parse_args()

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def prepare_training(train_dataloader):
    if args.unimodal == 'multimodal-tv':
        model = DGMaskAttnMultimodalModel1(args)
    elif args.unimodal == 'multimodal-vt':
        model = DGMaskAttnMultimodalModel2(args)
    elif args.unimodal == 'multimodal-tt':
        model = DGMaskAttnMultimodalModel3(args)
    print(args.unimodal)
    model.to(DEVICE)
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

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCHS)
    return model, optimizer, scheduler


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch):
    step = 0
    tr_loss = 0
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        sentence, input_ids, attention_mask, visual, visual_len, source_label, label_id, segment = batch
        input_ids, attention_mask, visual, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
        label_id = label_id.long()
        step += 1
        optimizer.zero_grad()
        loss_fct = CrossEntropyLoss()
        outputs = model(input_ids, attention_mask, visual, visual_len, label_id)
        # if args.unimodal == 'multimodal' or args.unimodal == 'multimodal-wogl' or args.unimodal == 'multimodal-noncondition' or args.unimodal == 'multimodal-sparse-tv' or args.unimodal == 'multimodal-sparse-vt':
        if epoch < args.warm_up:
            (logits, text_logits, visual_logits, text_sparse_loss, visual_sparse_loss, recon_loss, _, _) = outputs
            loss = loss_fct(logits, label_id.view(-1))
            text_loss = loss_fct(text_logits, label_id.view(-1))
            visual_loss = loss_fct(visual_logits, label_id.view(-1))
            total_loss = ALPHA * (text_sparse_loss) + BETA * (text_loss)
        else:
            (logits, text_logits, visual_logits, text_sparse_loss, visual_sparse_loss, recon_loss, _, _) = outputs
            loss = loss_fct(logits, label_id.view(-1))
            text_loss = loss_fct(text_logits, label_id.view(-1))
            visual_loss = loss_fct(visual_logits, label_id.view(-1))
            total_loss = loss + ALPHA * (text_sparse_loss + visual_sparse_loss) + BETA * (text_loss + visual_loss) + recon_loss      
        total_loss.backward()
        tr_loss += total_loss.item()
        optimizer.step()
        scheduler.step()
    tr_loss /= step
    return tr_loss

def eval_epoch(model, dev_dataloader, optimizer, domain=0, epoch=0):
    step = 0
    dev_loss = 0
    y_test = []
    preds = []
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, visual, visual_len, source_label, label_id, segment = batch
            input_ids, attention_mask, visual, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
            label_id = label_id.long()
            outputs = model(input_ids, attention_mask, visual, visual_len, label_id)
            # if args.unimodal == 'multimodal' or args.unimodal == 'multimodal-wogl' or args.unimodal == 'multimodal-noncondition':
            (logits, text_logits, visual_logits, text_sparse_loss, visual_sparse_loss, recon_loss, _, _) = outputs

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, label_id.view(-1))
            total_loss = loss
            # total_loss = loss + ALPHA * sparse_loss + BETA * unimodal_loss

            _, predicted = torch.max(logits.data, dim=1)
            y_test_val = label_id.view(-1).cpu().detach().tolist()
            preds_val = predicted.cpu().detach().tolist()
            y_test += y_test_val
            preds += preds_val
            # total_loss = loss
            dev_loss += total_loss.item()
        acc = accuracy_score(y_test, preds)
        dev_loss /= step
    return dev_loss, acc

def test_epoch(model: nn.Module, test_dataloader: DataLoader, domain=0, epoch=-1):
    model.eval()
    preds = []
    y_test = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, visual, visual_len, source_label, label_id, segment = batch
            input_ids, attention_mask, visual, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
            label_id = label_id.long()
            outputs = model(input_ids, attention_mask, visual, visual_len, label_id)
            # if args.unimodal == 'multimodal' or args.unimodal == 'multimodal-wogl' or args.unimodal == 'multimodal-noncondition':
            (logits, text_logits, visual_logits, text_sparse_loss, visual_sparse_loss, recon_loss, _, _) = outputs
            _, predicted = torch.max(logits.data, dim=1)
            y_test_val = label_id.view(-1).cpu().detach().tolist()
            preds_val = predicted.cpu().detach().tolist()
            y_test += y_test_val
            preds += preds_val
            # if domain == 1:
            #     print(preds_val)
            #     print(y_test_val)
        acc = accuracy_score(y_test, preds)
    return acc



def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False, epoch=-1, test=False, domain=0):
    acc = test_epoch(model, test_dataloader, epoch=epoch, domain=domain)
    return acc






def get_dataset():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_PATH)
    # tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH)
    # tokenizer = CLIPTokenizer.from_pretrained(PRETRAIN_PATH)

    if args.domain_type == 1: 
        TRAIN_DOMAIN = 'mosi/mosi_vgg.pkl' 
        TEST_DOMAIN1 = 'meld/meld_vgg.pkl'
        TEST_DOMAIN2 = 'mosei/mosei_vgg.pkl'
    elif args.domain_type == 2:
        TRAIN_DOMAIN = 'meld/meld_vgg.pkl'
        TEST_DOMAIN1 = 'mosi/mosi_vgg.pkl' 
        TEST_DOMAIN2 = 'mosei/mosei_vgg.pkl'
    elif args.domain_type == 3:
        TRAIN_DOMAIN = 'mosei/mosei_vgg.pkl'
        TEST_DOMAIN1 = 'mosi/mosi_vgg.pkl' 
        TEST_DOMAIN2 = 'meld/meld_vgg.pkl'



    train_data_path = os.path.join(PATH, TRAIN_DOMAIN)
    test_data_path1 = os.path.join(PATH, TEST_DOMAIN1)
    test_data_path2 = os.path.join(PATH, TEST_DOMAIN2)

    with open(train_data_path, "rb") as handle:
        train_domain = pickle.load(handle)

    with open(test_data_path1, "rb") as handle:
        test_domain1 = pickle.load(handle)

    with open(test_data_path2, "rb") as handle:
        test_domain2 = pickle.load(handle)


    train_domian_train_dataset = train_domain['train']
    train_domian_dev_dataset = train_domain['dev']
    train_domian_test_dataset = train_domain['test']

    
    test_domian1_test_dataset = test_domain1['test']
    test_domian2_test_dataset = test_domain2['test']

    if  'mosi' in TRAIN_DOMAIN:
        train_domain_train_data = MultimodalMOSIDataset(train_domian_train_dataset, tokenizer)
        train_domain_dev_data = MultimodalMOSIDataset(train_domian_dev_dataset, tokenizer)
        train_domain_test_data = MultimodalMOSIDataset(train_domian_test_dataset, tokenizer)
    elif 'mosei' in TRAIN_DOMAIN:
        train_domain_train_data = MultimodalMOSEIDataset(train_domian_train_dataset, tokenizer)
        train_domain_dev_data = MultimodalMOSEIDataset(train_domian_dev_dataset, tokenizer)
        train_domain_test_data = MultimodalMOSEIDataset(train_domian_test_dataset, tokenizer)
    elif 'meld' in TRAIN_DOMAIN:
        train_domain_train_data = MultimodalMeldDataset(train_domian_train_dataset, tokenizer)
        train_domain_dev_data = MultimodalMeldDataset(train_domian_dev_dataset, tokenizer)
        train_domain_test_data = MultimodalMeldDataset(train_domian_test_dataset, tokenizer)

    if  'mosei' in TEST_DOMAIN1:
        test_domain1_test_data = MultimodalMOSEIDataset(test_domian1_test_dataset, tokenizer)
    elif  'mosi' in TEST_DOMAIN1:
        test_domain1_test_data = MultimodalMOSIDataset(test_domian1_test_dataset, tokenizer)
    elif 'meld' in TEST_DOMAIN1:
        test_domain1_test_data = MultimodalMeldDataset(test_domian1_test_dataset, tokenizer)
        


    if 'mosei' in TEST_DOMAIN2:
        test_domain2_test_data = MultimodalMOSEIDataset(test_domian2_test_dataset, tokenizer)
    elif 'mosi' in TEST_DOMAIN2:
        test_domain2_test_data = MultimodalMOSIDataset(test_domian2_test_dataset, tokenizer)
    elif 'meld' in TEST_DOMAIN2:
        test_domain2_test_data = MultimodalMeldDataset(test_domian2_test_dataset, tokenizer)


    return train_domain_train_data, train_domain_dev_data, train_domain_test_data, test_domain1_test_data, test_domain2_test_data




def get_dataloader(train_domain_train_data, train_domain_dev_data, train_domain_test_data, test_domain1_test_data, test_domain2_test_data):
    train_domain_train_dataloader = DataLoader(train_domain_train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    train_domain_dev_dataloader = DataLoader(train_domain_dev_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    train_domain_test_dataloader = DataLoader(train_domain_test_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)

    test_domain1_test_dataloader = DataLoader(test_domain1_test_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    test_domain2_test_dataloader = DataLoader(test_domain2_test_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)

    num_train_optimization_steps = 0
    return train_domain_train_dataloader, train_domain_dev_dataloader, train_domain_test_dataloader, test_domain1_test_dataloader, test_domain2_test_dataloader


def train(
    model,
    train_domain_train_dataloader,
    train_domain_validation_dataloader,
    train_domain_test_dataloader,
    test_domian1_test_dataloader,
    test_domian2_test_dataloader,
    optimizer,
    scheduler,
):
    valid_losses, valid_acc = [], []
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_domain_train_dataloader, optimizer, scheduler, epoch=epoch_i)
        valid_loss, val_acc = eval_epoch(model, train_domain_validation_dataloader, optimizer, epoch=epoch_i)
        acc_train_domain = test_score_model(model, train_domain_test_dataloader, epoch=epoch_i)
        acc_domain1 = test_score_model(model, test_domian1_test_dataloader, domain=1, epoch=epoch_i)
        acc_domain2  = test_score_model(model, test_domian2_test_dataloader, domain=2, epoch=epoch_i)

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, val_acc:{} test_train_domain:{} test_domain1:{} test_domain2:{}".format(
                epoch_i, train_loss, valid_loss, val_acc, acc_train_domain, acc_domain1, acc_domain2
            )
        )
        valid_losses.append(valid_loss)
        valid_acc.append(val_acc)
        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "best_valid_loss": min(valid_losses),
                    "valid_loss": valid_loss,
                    "valid_acc": val_acc,
                    "best_valid_acc": max(valid_acc),
                    'acc_train_domain':acc_train_domain,
                    "acc_domain_1": acc_domain1,
                    "acc_domain2": acc_domain2,
                }
            )
        )



def main():
    wandb.init(project="DG-S-2-Stage", name=args.wandb_name)
    wandb.config.update(args)
    # args.seed = 42
    set_random_seed(args.seed)
    # full setting
    train_domain_train_data, train_domain_dev_data, train_domain_test_data, test_domain1_test_data, test_domain2_test_data = get_dataset()
    train_domain_train_dataloader, train_domain_dev_dataloader, train_domain_test_dataloader, test_domain1_test_dataloader, test_domain2_test_dataloader = get_dataloader(train_domain_train_data, train_domain_dev_data, train_domain_test_data, test_domain1_test_data, test_domain2_test_data)
    model, optimizer, scheduler = prepare_training(train_domain_train_dataloader)
    train(
        model=model,
        train_domain_train_dataloader=train_domain_train_dataloader,
        train_domain_validation_dataloader=train_domain_dev_dataloader,
        train_domain_test_dataloader=train_domain_test_dataloader,
        test_domian1_test_dataloader=test_domain1_test_dataloader,
        test_domian2_test_dataloader=test_domain2_test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler
        )



if __name__ == "__main__":
    main()


