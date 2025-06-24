import os
import re
import sys
sys.path.append('/home/gouqiaolin/model/LaMGen')
import pickle
import torch
import argparse
import random
import logging
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader
from torch import distributions
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.early_stop.pytorchtools import EarlyStopping
from model.lamgen_model import LaMGen_dual
from utils.bert_tokenizer import ExpressionBertTokenizer
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as hooks
from datetime import timedelta

br_re = re.compile('Br')
cl_re = re.compile('Cl')
smiles_token_re = re.compile('(\[[^\[\]]{1,6}\])')

reverse_br_re = re.compile('R')
reverse_cl_re = re.compile('L')


Ada_config = GPT2Config(
    architectures=["GPT2LMHeadModel"],
    model_type="GPT2LMHeadModel",
    vocab_size=836,
    n_positions=1800,
    n_ctx=380,
    n_embd=768,
    n_layer=12,
    n_head=8,

    task_specific_params={
        "text-generation": {
            "do_sample": True,
            "max_length": 380
        }
    }
)


def read_data(path):
    data = []
    with open(path, 'rb') as f:
        while True:
            try:
                aa = pickle.load(f)
                data.extend(aa)
            except EOFError:
                break
    return data

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="Pretrained_model", type=str, help='')
    parser.add_argument('--vocab_path', default="./data/torsion_voc.csv", type=str, help='')
    parser.add_argument('--every_step_save_path', default="./checkpoint/finetuning", type=str, help='')
    parser.add_argument('--early_stop_path', default="./checkpoint/finetuning", type=str, help='')
    parser.add_argument('--batch_size', default=12, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=15, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=20000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-2, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    parser.add_argument('--seed', default=2024, type=int, required=False, help='print log steps')
    return parser.parse_args()

def get_all_normal_dis_pdf(voc_len=836, confs_num=629):
    means = torch.arange(1, confs_num + 1).float()
    std_dev = 2.0
    normal_dist_list = [distributions.Normal(mean, std_dev) for mean in means]

    pdf_list = []

    zero_pdf = torch.zeros(voc_len)
    zero_pdf[0] = 1.0
    pdf_list.append(zero_pdf)

    for idx, normal_dist in enumerate(normal_dist_list):
        pdf = torch.zeros(voc_len)

        pdf[1:confs_num + 1] = normal_dist.log_prob(means).exp()

        pdf[idx + 1] = pdf[idx + 1] * 2

        normalized_pdf = pdf / pdf.sum()

        pdf_list.append(normalized_pdf)

    pdf_tensor = torch.stack(pdf_list)

    return pdf_tensor


def calculate_loss_and_accuracy_confs(outputs, labels, device):
    pdf_tensor = get_all_normal_dis_pdf().to(device)

    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    shift_labels_copy = shift_labels.masked_fill((shift_labels != 829), 0)
    shift_labels = shift_labels.masked_fill((shift_labels >= 630), 0)
    shift_labels_copy_copy = shift_labels.clone()

    shift_labels = shift_labels + shift_labels_copy
    one_hot = F.one_hot(shift_labels, num_classes=836).float()

    non_zero_indices = torch.nonzero(shift_labels_copy_copy, as_tuple=False)

    if non_zero_indices.size(0) > 0:
        rows = non_zero_indices[:, 0]
        cols = non_zero_indices[:, 1]
        conf_ids = shift_labels[rows, cols]

        poisson_one_hot = pdf_tensor[conf_ids]
        one_hot[rows, cols] = poisson_one_hot

    logsoftmax = F.log_softmax(shift_logits, dim=-1)

    not_ignore = shift_labels.ne(0)
    one_hot = not_ignore.unsqueeze(-1) * one_hot

    loss = -torch.sum(one_hot * logsoftmax)

    _, preds = shift_logits.max(dim=-1)
    num_targets = not_ignore.long().sum().item()
    correct = ((shift_labels == preds) & not_ignore).float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy


def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits

    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()      # (B, L-1, V)
    shift_labels = labels[..., 1:].contiguous().to(device)  # (B, L-1)

    mask = (shift_labels >= 630)
    shift_labels = shift_labels.masked_fill(~mask, 0)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    preds = shift_logits.argmax(dim=-1)  # (B, L-1)

    not_ignore = shift_labels.ne(0)
    num_targets = not_ignore.sum()

    correct = ((preds == shift_labels) & not_ignore).sum()

    accuracy = correct.float() / (num_targets.float() + 1e-8)
    loss = loss / (num_targets.float() + 1e-8)

    return loss, accuracy


class Cluster_Dataset(Dataset):

    def __init__(self, data_df, tokenizer, indices=None):

        protein1 = data_df['target_id1']
        protein2 = data_df['target_id2']

        self.smiles = data_df['SMILES']
        self.geos = data_df['geos']

        self.tokenizer = tokenizer
        self.protein_dir = '/home/gouqiaolin/dataset/Papyrus/ESMC'

        if indices is not None:

            protein_list1 = []
            protein_list2 = []

            for idx in indices:

                tar1 = protein1[idx]
                tar2 = protein2[idx]

                if not tar1.endswith("_WT"):
                    tar1 += "_WT"
                if not tar2.endswith("_WT"):
                    tar2 += "_WT"

                npy1 = os.path.join(self.protein_dir, f"{tar1}.npy")
                npy2 = os.path.join(self.protein_dir, f"{tar2}.npy")

                protein_list1.append(npy1)
                protein_list2.append(npy2)

            self.protein_list1 = protein_list1
            self.protein_list2 = protein_list2

        else:

            print('$$$$$$$$$$$$ You should input index of dataset $$$$$$$$$$$$')

    def __getitem__(self, index):
        # mol = []
        geo = self.geos[index]
        geo = geo.replace("'", "").replace("[", "").replace("]", "").replace(",", "").strip()
        tok = tokenize(self.smiles[index])
        ligand = tok + " " + geo

        ligand = reverse_br_re.sub('Br', ligand)
        ligand = reverse_cl_re.sub('Cl', ligand)

        protein_path1 = self.protein_list1[index]
        protein_path2 = self.protein_list2[index]

        protein1 = np.load(protein_path1, allow_pickle=True)
        protein2 = np.load(protein_path2, allow_pickle=True)

        ligand = '<|beginoftext|> <|mask:0|> <|mask:0|> ' + str(ligand) + ' <|endofmask|>'
        mol = [self.tokenizer.encode(ligand, truncation=False, max_length=200, return_special_tokens_mask=True,
                                add_special_tokens=False)]

        mol.append(protein1)
        mol.append(protein2)

        return mol

    def __len__(self):
        return len(self.protein_list1)

    def collate_fn(self, mix_batch):
        batch, protein_batch1, protein_batch2 = list(zip(*mix_batch))
        input_ids = []

        input_lens_list = [len(w) for w in batch]
        input_protein_len_list1 = [len(ww) for ww in protein_batch1]
        input_protein_len_list2 = [len(ww) for ww in protein_batch2]

        max_input_len = max(input_lens_list)
        max_protein_len = max(max(input_protein_len_list1), max(input_protein_len_list2))  # 蛋白最长序列长度

        protein_ids1 = np.zeros((len(protein_batch1), max_protein_len, len(protein_batch1[0][0])),
                                dtype=protein_batch1[0][0].dtype)
        protein_ids2 = np.zeros((len(protein_batch2), max_protein_len, len(protein_batch2[0][0])),
                                dtype=protein_batch2[0][0].dtype)

        for btc_idx in range(len(batch)):
            input_len = len(batch[btc_idx])
            input_ids.append(batch[btc_idx])
            input_ids[btc_idx].extend([self.tokenizer.pad_token_id] * (max_input_len - input_len))

            protein_ids1[btc_idx, :len(protein_batch1[btc_idx]), :] = protein_batch1[btc_idx]
            protein_ids2[btc_idx, :len(protein_batch2[btc_idx]), :] = protein_batch2[btc_idx]

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(protein_ids1, dtype=torch.float32),
                torch.tensor(protein_ids2, dtype=torch.float32))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '22227'
    dist.init_process_group("nccl", rank=rank, timeout=timedelta(seconds=7200), world_size=world_size)
    rank = dist.get_rank()
    ws = dist.get_world_size()
    return rank, ws


def set_dataset(args):

    csv_path = './data/ft_dataset.csv'
    data_df = pd.read_csv(csv_path)
    data_length = len(data_df)

    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    val_num = int(0.1 * data_length)

    val_inds = np.random.choice(range(0, data_length), val_num, replace=False)
    train_inds = np.setdiff1d(range(0, data_length), val_inds)

    train_dataset = Cluster_Dataset(data_df, tokenizer, train_inds)
    val_dataset = Cluster_Dataset(data_df, tokenizer, val_inds)

    return train_dataset, val_dataset


def tokenize(smiles):
    smiles = br_re.sub('R', smiles)
    smiles = cl_re.sub('L', smiles)
    tokens = smiles_token_re.split(smiles)
    tokenized = (token if token.startswith('[') else ' '.join(token) for token in tokens if token)
    return ' '.join(tokenized) + ' GEO'


def init_logging(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if rank == 0:
        file_handler = logging.FileHandler('./ft_output.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)


def train(rank, args, world_size):

    init_logging(rank)

    rank, world_size = setup(rank, world_size)
    torch.cuda.set_device(rank)

    if world_size > 1:
        dist.barrier()
    device = torch.device("cuda", rank)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    train_set, val_set = set_dataset(args)

    if world_size <= 1:
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=train_set.collate_fn,
                                  pin_memory=True)

        val_loader = DataLoader(dataset=val_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=val_set.collate_fn,
                                pin_memory=True)

    else:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(train_set, num_workers=0, batch_size=args.batch_size, sampler=train_sampler, collate_fn=train_set.collate_fn, pin_memory=True)

        val_sampler = DistributedSampler(val_set, shuffle=False)
        val_loader = DataLoader(val_set, num_workers=0, batch_size=args.batch_size, sampler=val_sampler, collate_fn=val_set.collate_fn, pin_memory=True)

    num_training_steps = args.epochs * len(train_loader)
    model = LaMGen_dual(pretrain_path=args.model_path, config=Ada_config)

    state_dict = torch.load("./checkpoint/dual/ckpt_8", map_location='cuda')
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)

    model.to(rank)

    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        model.register_comm_hook(state=None, hook=hooks.fp16_compress_hook)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    batch_steps = 0

    for epoch in tqdm(range(args.epochs)):

        if world_size > 1:
            train_sampler.set_epoch(epoch)

        model.train()

        for mix_batch in tqdm(train_loader):
            try:
                batch, protein_batch1, protein_batch2 = mix_batch
                batch_steps += 1
                batch = batch.to(device, non_blocking=True)
                protein_batch1 = protein_batch1.to(device, non_blocking=True)
                protein_batch2 = protein_batch2.to(device, non_blocking=True)

                outputs = model(batch, protein_batch1, protein_batch2)

                loss_conf, acc_conf = calculate_loss_and_accuracy_confs(outputs, batch, device)
                loss_smiles, acc_smiles = calculate_loss_and_accuracy(outputs, batch, device)

                loss = loss_conf*0.2 + loss_smiles*0.8
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if rank == 0 and batch_steps % args.log_step == 0:
                    logging.info(f"train epoch {epoch}/{args.epochs}, batch {batch_steps}/{num_training_steps}, "
                                 f"loss {loss.item()}, confs_accuracy {acc_conf}, smi_accuracy {acc_smiles}")

                del batch, protein_batch1, protein_batch2, outputs, loss
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logging.warning(f"OOM at step {batch_steps}, skipping batch...")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

        if rank == 0 and epoch >= 0:
            torch.save(model.state_dict(), f"./checkpoint/finetuning/ft_ckpt_{epoch}")

        if rank == 0:
            evaluate(model, val_loader, args=args)


def evaluate(model, dataloader, args):

    device = torch.device(f"cuda:{dist.get_rank()}") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()
    loss_list, acc_list = [], []
    conf_loss_list, conf_acc_list = [], []
    smi_loss_list, smi_acc_list = [], []
    batch_steps = 0
    early_stopping = EarlyStopping(patience=10, verbose=False)

    with torch.no_grad():
        for mix_batch in dataloader:
            batch, protein_batch1, protein_batch2 = mix_batch
            batch_steps += 1
            batch = batch.to(device)
            protein_batch1 = protein_batch1.to(device)
            protein_batch2 = protein_batch2.to(device)

            outputs = model(batch, protein_batch1, protein_batch2)

            # conf loss
            loss_conf, acc_conf = calculate_loss_and_accuracy_confs(outputs, batch, device)
            # smiles loss
            loss_smiles, acc_smiles = calculate_loss_and_accuracy(outputs, batch, device)

            conf_loss_list.append(float(loss_conf))
            conf_acc_list.append(float(acc_conf))

            smi_loss_list.append(float(loss_smiles))
            smi_acc_list.append(float(acc_smiles))

            total_loss = (loss_conf + loss_smiles) / 2
            total_acc = (acc_conf + acc_smiles) / 2

            loss_list.append(float(total_loss))
            acc_list.append(float(total_acc))

            del batch, protein_batch1, protein_batch2, outputs
            torch.cuda.empty_cache()

    epoch_loss = np.mean(loss_list)
    early_stopping(epoch_loss, model, args.early_stop_path)

    logging.info(f"Total loss: {np.mean(loss_list):.4f}, Total acc: {np.mean(acc_list):.4f}")
    logging.info(f"Confs loss: {np.mean(conf_loss_list):.4f}, Confs acc: {np.mean(conf_acc_list):.4f}")
    logging.info(f"SMILES loss: {np.mean(smi_loss_list):.4f}, SMILES acc: {np.mean(smi_acc_list):.4f}")


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    args.model_path = './Pretrained_model/RTM_torsion_countinue_v2_epoch7'
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)

    world_size = 2
    print('Let\'s use', world_size, 'GPUs!')

    print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))

    if world_size > 1:

        mp.spawn(train,
                 args=(args, world_size),
                 nprocs=world_size,
                 join=True)

    else:

        train(0, args, world_size)


