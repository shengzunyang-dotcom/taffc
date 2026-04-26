from lib2to3.pgen2 import token
from PIL import Image
import torch
import wandb
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
from sklearn.metrics import accuracy_score, f1_score
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.electra.tokenization_electra import ElectraTokenizer
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score

# from src.models import *
# from src.models_v1 import *
# from src.models_dg import *

from src.models import *


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

parser.add_argument("--warm_up", type=int, default=5)

parser.add_argument("--test", type=int, default=0)


parser.add_argument("--t_dim", type=int, default=768)
parser.add_argument("--v_dim", type=int, default=512)
parser.add_argument("--a_dim", type=int, default=1024)

parser.add_argument("--dg_label_dim", type=int, default=3)
parser.add_argument("--ds_label_dim", type=int, default=1)




args = parser.parse_args()

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
    model = KnowledgeInjectionModel(args)


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

def compute_accurracy(preds, y_test):
    preds = np.array(preds)
    y_test = np.array(y_test)

    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(y_test, a_min=-2., a_max=2.)
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    acc5 = multiclass_acc(test_preds_a5, test_truth_a5)

    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])
    preds = preds[non_zeros]
    y_test = y_test[non_zeros]
    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    preds = preds >= 0
    y_test = y_test >= 0
    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)
    return acc, mae, corr, f_score, acc5, acc7


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch):
    step = 0
    tr_loss = 0
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        sentence, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, source_label, label_id, segment = batch
        input_ids, attention_mask, visual, visual_mask, audio, audio_mask, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), visual_mask.to(DEVICE), audio.to(DEVICE), audio_mask.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
        src_label = source_label.long()
        label_id = label_id.long()
        step += 1
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)

        dg_loss_fct, ds_loss_fct = get_loss_func()        
        # if args.unimodal == 'multimodal' or args.unimodal == 'multimodal-wogl' or args.unimodal == 'multimodal-noncondition' or args.unimodal == 'multimodal-noncondition' or args.unimodal == 'multimodal-sparse-tv' or args.unimodal == 'multimodal-sparse-vt':
        if epoch < args.warm_up:
            (text_logits, visual_logits, audio_logits) = outputs
            text_loss = dg_loss_fct(text_logits, label_id.view(-1))
            visual_loss = dg_loss_fct(visual_logits, label_id.view(-1))
            audio_loss = dg_loss_fct(audio_logits, label_id.view(-1))
            total_loss = text_loss + audio_loss + visual_loss
        else:
            (logits) = outputs
            loss = ds_loss_fct(logits, src_label.view(-1))
            loss = ds_loss_fct(text_logits, label_id.view(-1))
            total_loss = loss

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
    text_preds = []
    visual_preds = []
    audio_preds = []

    preds = []
    labels = []
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, source_label, label_id, segment = batch
            input_ids, attention_mask, visual, visual_mask, audio, audio_mask, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), visual_mask.to(DEVICE), audio.to(DEVICE), audio_mask.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
            source_label = source_label.long()
            label_id = label_id.long()
            outputs = model(input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)
            dg_loss_fct, ds_loss_fct = get_loss_func()

            if epoch < args.warm_up:
                (text_logits, visual_logits, audio_logits) = outputs
                text_loss = dg_loss_fct(text_logits, label_id.view(-1))
                visual_loss = dg_loss_fct(visual_logits, label_id.view(-1))
                audio_loss = dg_loss_fct(audio_logits, label_id.view(-1))
                _, text_predicted = torch.max(text_logits.data, dim=1)
                _, visual_predicted = torch.max(visual_logits.data, dim=1)
                _, audio_predicted = torch.max(audio_logits.data, dim=1)

                y_test_val = label_id.view(-1).cpu().detach().tolist()
                
                text_preds_val = text_predicted.cpu().detach().tolist()
                text_preds += text_preds_val
                visual_preds_val = visual_predicted.cpu().detach().tolist()
                visual_preds += visual_preds_val
                audio_preds_val = audio_predicted.cpu().detach().tolist()
                audio_preds += audio_preds_val
                y_test += y_test_val
                
                total_loss = text_loss + audio_loss + visual_loss
            else:
                (logits) = outputs
                loss = ds_loss_fct(logits, source_label.view(-1))
                total_loss = loss
                if args.domain_type == 1 or args.domain_type == 2:
                    source_label = source_label.detach().cpu().numpy()
                    source_label = np.squeeze(source_label).tolist()
                    logits = logits.detach().cpu().numpy()
                    logits = np.squeeze(logits).tolist()
                    preds.extend(logits)
                    labels.extend(label_ids)
                else:
                    (logits) = outputs
                    _, predicted = torch.max(logits.data, dim=1)
                    y_test_val = label_id.view(-1).cpu().detach().tolist()
                    preds_val = predicted.cpu().detach().tolist()
                    labels += y_test_val
                    preds += preds_val
            dev_loss += total_loss.item()

        dev_loss /= step
        if epoch < args.warm_up:
            text_acc = accuracy_score(y_test, text_preds)
            visual_acc = accuracy_score(y_test, visual_preds)
            audio_acc = accuracy_score(y_test, audio_preds)
            return dev_loss, text_acc, visual_acc, audio_acc
        else:
            if args.domain_type == 1 or args.domain_type == 2:
                acc, mae, corr, f_score, acc5, acc7 = compute_accurracy(preds, labels)
                return dev_loss, acc, mae, corr, f_score, acc5, acc7
            else:
                acc = accuracy_score(y_test, preds)
                return dev_loss, acc




def test_epoch(model: nn.Module, test_dataloader: DataLoader, domain=0, epoch=-1):
    model.eval()
    preds = []
    y_test = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, source_label, label_id, segment = batch
            input_ids, attention_mask, visual, visual_mask, audio, audio_mask, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), visual_mask.to(DEVICE), audio.to(DEVICE), audio_mask.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
            label_id = label_id.long()
            outputs = model(input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id)
            # if args.unimodal == 'multimodal' or args.unimodal == 'multimodal-wogl' or args.unimodal == 'multimodal-noncondition':
            (logits, text_logits, visual_logits, text_sparse_loss, visual_sparse_loss, recon_loss) = outputs
            _, predicted = torch.max(logits.data, dim=1)
            y_test_val = label_id.view(-1).cpu().detach().tolist()
            preds_val = predicted.cpu().detach().tolist()
            y_test += y_test_val
            preds += preds_val
        acc = accuracy_score(y_test, preds)
    return acc

def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False, epoch=-1, test=False, domain=0):
    acc = test_epoch(model, test_dataloader, epoch=epoch, domain=domain)
    return acc






def get_dataset():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_PATH)
    # tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH)
    # tokenizer = CLIPTokenizer.from_pretrained(PRETRAIN_PATH)
    mosi_path = 'merge/mosi_vgg_hubert.pkl'
    mosei_path = 'merge/mosei_vgg_hubert.pkl'
    meld_path = 'merge/meld_vgg_hubert.pkl'

    mosi_data_path = os.path.join(PATH, mosi_path)
    mosei_data_path = os.path.join(PATH, mosei_path)
    meld_data_path = os.path.join(PATH, meld_path)

    with open(mosi_data_path, "rb") as handle:
        mosi_data = pickle.load(handle)

    with open(mosei_data_path, "rb") as handle:
        mosei_data = pickle.load(handle)

    with open(meld_data_path, "rb") as handle:
        meld_data = pickle.load(handle)


    if args.domain_type == 1:
        train_dataset = mosi_data['train']
        dev_dataset = mosi_data['dev']
        test_dataset = mosi_data['test']
        knowledge_train_dataset = mosei_data['train'] + meld_data['train'] 
        knowledge_test_dataset = mosi_data['test'] + meld_data['test']
        if args.test == 0:
            knowledge_train_dataset = knowledge_train_dataset[0: 80]
            knowledge_test_dataset = knowledge_test_dataset[0: 80]

    elif args.domain_type == 2:
        train_dataset = mosei_data['train']
        dev_dataset = mosei_data['dev']
        test_dataset = mosei_data['test']
        knowledge_train_dataset = mosi_data['train'] + meld_data['train']
        knowledge_test_dataset = mosi_data['test'] + meld_data['test']
    elif args.domain_type == 3:
        train_dataset = meld_data['train']
        dev_dataset = meld_data['dev']
        test_dataset = meld_data['test']
        knowledge_train_dataset = mosi_data['train'] + mosei_data['train']
        knowledge_test_dataset =  mosi_data['test'] + mosei_data['test']

    train_data = MultimodalDataset(train_dataset, tokenizer)
    dev_data = MultimodalDataset(dev_dataset, tokenizer)
    test_data = MultimodalDataset(test_dataset, tokenizer)
    knowledge_train_data = MultimodalDataset(knowledge_train_dataset, tokenizer)
    knowledge_test_data = MultimodalDataset(knowledge_test_dataset, tokenizer)

    return train_data, dev_data, test_data, knowledge_train_data, knowledge_test_data




def get_dataloader(
        train_data, 
        dev_data, 
        test_data, 
        knowledge_train_data,
        knowledge_test_data, 
        ):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    knowledge_train_dataloader = DataLoader(knowledge_train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    knowledge_test_dataloader = DataLoader(knowledge_test_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)

    num_train_optimization_steps = 0
    return train_dataloader, dev_dataloader, test_dataloader, knowledge_train_dataloader, knowledge_test_dataloader


def train(
    model,
    train_dataloader,
    dev_dataloader,
    test_dataloader,
    knowledge_train_dataloader,
    knowledge_test_dataloader,
    optimizer,
    scheduler,
):
    valid_losses, valid_accs = [], []
    mosi_best_accs, mosei_best_accs, meld_best_accs = [], [], []
    for epoch in range(int(args.n_epochs)):
        if epoch < args.warm_up:
            train_loss = train_epoch(model, knowledge_train_dataloader, optimizer, scheduler, epoch)
            knowledge_loss, knowledge_text_acc, knowledge_visual_acc, knowledge_audio_acc = eval_epoch(model, knowledge_test_dataloader, optimizer, epoch=epoch)
        else:
            train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch)
            if args.domain_type == 0 or args.domain_type == 1:
                dev_loss, dev_acc, dev_mae, dev_corr, dev_f_score, dev_acc5, dev_acc7 = eval_epoch(model, dev_dataloader, optimizer, epoch=epoch)
                test_loss, test_acc, test_mae, test_corr, test_f_score, test_acc5, test_acc7 = eval_epoch(model, test_dataloader, epoch=epoch)
            else:
                dev_loss, dev_acc = eval_epoch(model, dev_dataloader, optimizer, epoch=epoch)
                test_loss, test_acc = eval_epoch(model, test_dataloader, epoch=epoch) 


        # print(
        #     f"epoch:{epoch_i}, train_loss1:{train_loss}, valid_loss:{valid_loss}, valid_acc:{valid_acc} mosi_acc:{test_mosi_acc} mosei_acc:{test_mosei_acc} meld_acc:{test_meld_acc}"
        # )
        # valid_losses.append(valid_loss)
        # valid_accs.append(valid_acc)


        # mosi_best_accs.append(test_mosi_acc)
        # mosei_best_accs.append(test_mosei_acc)
        # meld_best_accs.append(test_meld_acc)

        # wandb.log(
        #     (
        #         {
        #             "train_loss1": train_loss,
        #             "best_valid_loss1": min(valid_losses),
        #             "valid_loss": valid_loss,
        #             "valid_acc": valid_acc,
        #             "best_valid_acc": max(valid_accs),


        #             'test_mosi_acc': test_mosi_acc,
        #             "test_mosei_acc": test_mosei_acc,
        #             "test_meld_acc": test_meld_acc,
        #             'mosi_best_acc': max(mosi_best_accs),
        #             "mosei_best_acc": max(mosei_best_accs),
        #             "meld_best_acc": max(meld_best_accs),

        #             "keep_ratio_text": text_keep_ratio,
        #             "keep_ratio_visual": visual_keep_ratio,
        #             "diff_ratio": different_ratio,
        #         }
        #     )
        # )



def main():
    # wandb.init(project="knowledge-injection", name=args.wandb_name)
    # wandb.config.update(args)
    # args.seed = 42
    set_random_seed(args.seed)
    # full setting
    train_data, dev_data, test_data, knowledge_train_data, knowledge_test_data = get_dataset()
    train_dataloader, dev_dataloader, test_dataloader, knowledge_train_dataloader, knowledge_test_dataloader  = get_dataloader(train_data, dev_data, test_data, knowledge_train_data, knowledge_test_data)
    model, optimizer, scheduler = prepare_training(train_dataloader)
    train(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        test_dataloader=test_dataloader,
        knowledge_train_dataloader=knowledge_train_dataloader,
        knowledge_test_dataloader=knowledge_test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler
        )



if __name__ == "__main__":
    main()


