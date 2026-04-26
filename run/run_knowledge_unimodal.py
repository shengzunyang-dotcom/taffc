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
from src.model_uni import UnimodalModelText, UnimodalModelVisual, UnimodalModelAudio, UnimodalModelJoint, DGUnimodalText, DGUnimodalVisual, DGUnimodalAcoustic

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

parser.add_argument("--warm_up", type=int, default=3)

parser.add_argument("--test", type=int, default=0)


parser.add_argument("--t_dim", type=int, default=768)
parser.add_argument("--v_dim", type=int, default=512)
parser.add_argument("--a_dim", type=int, default=1024)

parser.add_argument("--dg_label_dim", type=int, default=3)
parser.add_argument("--ds_label_dim", type=int, default=1)

parser.add_argument("--step", type=int, default=1)

parser.add_argument("--dsbert", type=str, default='t5')
parser.add_argument("--dgbert", type=str, default='bert')
parser.add_argument("--checkpoint_dir", type=str, default='checkpoint-bert')


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
    if args.unimodal == 'text':
        model = UnimodalModelText(args)
    elif args.unimodal == 'visual':
        model = UnimodalModelVisual(args)
    elif args.unimodal == 'audio':
        model = UnimodalModelAudio(args)
    elif args.unimodal == 'dgtext':
        model = DGUnimodalText(args)
    elif args.unimodal == 'dgvisual':
        model = DGUnimodalVisual(args)
    elif args.unimodal == 'dgaudio':
        model = DGUnimodalAcoustic(args)
    else:
        print('None')
        exit(0)
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

def compute_accurracy(preds, y_test, use_zero=False):
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
        
        label_id = label_id.long()
        step += 1
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id, epoch)

        dg_loss_fct, ds_loss_fct = get_loss_func()        

        (logits) = outputs
        loss = ds_loss_fct(logits.view(-1), source_label.view(-1))
        total_loss = loss
        total_loss.backward()
        tr_loss += total_loss.item()
        optimizer.step()
        scheduler.step()

    tr_loss /= step
    return tr_loss

def eval_epoch(model, dev_dataloader, optimizer, domain=0, epoch=0, test=False):
    step = 0
    dev_loss = 0
    y_test = []

    text_preds = []
    visual_preds = []
    audio_preds = []
    
    segments = []
    sentences = []

    preds = []
    labels = []
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            sentence, input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, source_label, label_id, segment = batch
            input_ids, attention_mask, visual, visual_mask, audio, audio_mask, source_label, label_id = input_ids.to(DEVICE), attention_mask.to(DEVICE), visual.to(DEVICE), visual_mask.to(DEVICE), audio.to(DEVICE), audio_mask.to(DEVICE), source_label.to(DEVICE), label_id.to(DEVICE)
            # source_label = source_label.long()
            label_id = label_id.long()
            outputs = model(input_ids, attention_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_id)
            dg_loss_fct, ds_loss_fct = get_loss_func()

            (logits) = outputs
            loss = ds_loss_fct(logits.view(-1), source_label.view(-1))

            source_label = source_label.detach().cpu().numpy()
            source_label = np.squeeze(source_label).tolist()
            logits = logits.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            preds.extend(logits)
            labels.extend(source_label)
            total_loss = loss
            dev_loss += total_loss.item()

            segments.extend(segment)
            sentences.extend(sentence)

        dev_loss /= step
        acc, mae, corr, f_score, acc5, acc7 = compute_accurracy(preds, labels)

        if test:
            file_obj = open(f'./r-case/{epoch}-{args.unimodal}-{args.domain_type}.txt', 'w')
            for i in range(len(preds)):
                line = f'{segments[i]} ____ {sentences[i]} ____ {labels[i]} ____ {preds[i]}' + '\n'
                file_obj.writelines(line)
            file_obj.close()
    return dev_loss, acc, mae, corr, f_score, acc5, acc7




def get_dataset():
    if args.dsbert == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAIN_PATH)
    elif args.dsbert == 'electra':
        tokenizer = AutoTokenizer.from_pretrained(ELECTRA_PRETRAIN_PATH)
    elif args.dsbert == 't5':
        tokenizer = AutoTokenizer.from_pretrained(T5_PRETRAIN_PATH)

    # tokenizer = BertTokenizer.from_pretrained(PRETRAIN_PATH)
    # tokenizer = CLIPTokenizer.from_pretrained(PRETRAIN_PATH)
    mosi_path = 'merge/mosi_vgg_hubert.pkl'
    mosei_path = 'merge/mosei_vgg_hubert.pkl'
    meld_path = 'merge/meld_vgg_hubert.pkl'

    mosi_data_path = os.path.join(PATH, mosi_path)
    mosei_data_path = os.path.join(PATH, mosei_path)
    meld_data_path = os.path.join(PATH, meld_path)

    if args.domain_type == 1:
        with open(mosi_data_path, "rb") as handle:
            mosi_data = pickle.load(handle)
        train_dataset = mosi_data['train']
        dev_dataset = mosi_data['dev']
        test_dataset = mosi_data['test']


    elif args.domain_type == 2:
        with open(mosei_data_path, "rb") as handle:
            mosei_data = pickle.load(handle)
        train_dataset = mosei_data['train']
        dev_dataset = mosei_data['dev']
        test_dataset = mosei_data['test']


    elif args.domain_type == 3:
        with open(meld_data_path, "rb") as handle:
            meld_data = pickle.load(handle)
        train_dataset = meld_data['train']
        dev_dataset = meld_data['dev']
        test_dataset = meld_data['test']

    train_data = MultimodalDataset(train_dataset, tokenizer)
    dev_data = MultimodalDataset(dev_dataset, tokenizer)
    test_data = MultimodalDataset(test_dataset, tokenizer)


    return train_data, dev_data, test_data




def get_dataloader(
        train_data, 
        dev_data, 
        test_data, 
        ):
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=padding_collate_fn)


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
    valid_losses, valid_accs = [], []
    test_accs, test_f_scores, test_maes, test_corrs = [], [], [], []
    for epoch in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch)
        valid_loss, valid_acc, valid_mae, valid_corr, valid_f_score, valid_acc5, valid_acc7 = eval_epoch(model, dev_dataloader, optimizer, epoch=epoch)
        test_loss, test_acc, test_mae, test_corr, test_f_score, test_acc5, test_acc7 = eval_epoch(model, test_dataloader, optimizer, epoch=epoch, test=True)
    
        print(
            f"epoch:{epoch}, train_loss:{train_loss}, valid_loss:{valid_loss}, valid_acc:{valid_acc}, test_acc:{test_acc}"
        )
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        test_accs.append(test_acc)
        test_f_scores.append(test_f_score)
        test_maes.append(test_mae)
        test_corrs.append(test_corr)
        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "best_valid_loss": min(valid_losses),
                    "valid_loss": valid_loss,
                    'valid_acc': valid_acc,
                    "valid_best_acc": max(valid_accs),

                    'test_acc': test_acc,
                    "test_f_score": test_f_score,
                    "test_mae": test_mae,
                    'test_corr': test_corr,
                    "test_best_acc": max(test_accs),
                    "test_best_f_score": max(test_f_scores),
                    "test_best_mae": min(test_maes),
                    "test_best_corr": max(test_corrs),
                }
            )
        )



def main():
    wandb.init(project="knowledge-injection-bert", name=args.wandb_name)
    wandb.config.update(args)
    # args.seed = 42
    set_random_seed(args.seed)
    # full setting
    train_data, dev_data, test_data = get_dataset()
    train_dataloader, dev_dataloader, test_dataloader  = get_dataloader(train_data, dev_data, test_data)
    model, optimizer, scheduler = prepare_training(train_dataloader)
    train(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler
        )



if __name__ == "__main__":
    main()


