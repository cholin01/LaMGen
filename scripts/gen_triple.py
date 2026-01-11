import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.lamgen_model import LaMGen_triple
from utils.bert_tokenizer import ExpressionBertTokenizer
from train_triple import Ada_config


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
    parser.add_argument('--model_path', default="../checkpoint/triple/triple_target_ckpt", type=str, help='')
    parser.add_argument('--vocab_path', default="../data/torsion_voc.csv", type=str, help='')
    parser.add_argument('--output_path', default="../generation/egfr_her2_vgfr_10.csv", type=str, help='')
    parser.add_argument('--batch_size', default=50, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, required=False, help='epochs')
    return parser.parse_args()


def decode(matrix):
    chars = []
    for i in matrix:
        if i == '<|endofmask|>': break
        chars.append(i)
    seq = " ".join(chars)
    return seq


@torch.no_grad()
def predict(model, tokenizer, batch_size, protein1, protein2, protein3, seed,
            text="<|beginoftext|> <|mask:0|> <|mask:0|>"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, _ = load_model(args.save_model_path, args.vocab_path)
    # text = ""
    protein_batch1 = protein1
    protein_batch2 = protein2
    protein_batch3 = protein3

    model.to(device)
    model.eval()
    # time1 = time.time()
    max_length = 195
    input_ids = []
    input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    input_length = len(input_ids)

    input_tensor = torch.zeros(batch_size, input_length).long()
    input_tensor[:] = torch.tensor(input_ids)

    Seq_list = []

    finished = torch.zeros(batch_size, 1).byte().to(device)

    protein_batch1 = torch.tensor(protein_batch1, dtype=torch.float32)
    protein_batch1 = protein_batch1.to(device)
    protein_batch1 = protein_batch1.repeat(batch_size, 1, 1)

    protein_batch2 = torch.tensor(protein_batch2, dtype=torch.float32)
    protein_batch2 = protein_batch2.to(device)
    protein_batch2 = protein_batch2.repeat(batch_size, 1, 1)

    protein_batch3 = torch.tensor(protein_batch3, dtype=torch.float32)
    protein_batch3 = protein_batch3.to(device)
    protein_batch3 = protein_batch3.repeat(batch_size, 1, 1)

    for i in range(max_length):
        inputs = input_tensor.to(device)
        outputs = model(inputs, protein_batch1, protein_batch2, protein_batch3)
        logits = outputs.logits
        logits = F.softmax(logits[:, -1, :])

        last_token_id = torch.multinomial(logits, 1)
        # last_token_id = torch.argmax(logits,1).view(-1,1)

        EOS_sampled = (last_token_id == tokenizer.encode('<|endofmask|>', add_special_tokens=False))
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            print('End')
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)

        Seq_list.append(last_token)
    Seq_list = np.array(Seq_list).T

    return Seq_list


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load_protein():

    npy1 = '../ESMC_example/P42336_WT.npy'
    npy2 = '../ESMC_example/P31749_WT.npy'
    npy3 = '../ESMC_example/P42345_WT.npy'

    protein_list1 = []
    protein_list2 = []
    protein_list3 = []

    protein_1 = np.load(npy1, allow_pickle=True)
    protein_2 = np.load(npy2, allow_pickle=True)
    protein_3 = np.load(npy3, allow_pickle=True)

    max_dim = max(protein_1.shape[0], protein_2.shape[0], protein_3.shape[0])

    padding_tensor1 = max_dim - protein_1.shape[0]
    padding_tensor2 = max_dim - protein_2.shape[0]
    padding_tensor3 = max_dim - protein_3.shape[0]

    padded_protein1 = np.pad(protein_1, ((0, padding_tensor1), (0, 0)), mode='constant', constant_values=0)
    padded_protein2 = np.pad(protein_2, ((0, padding_tensor2), (0, 0)), mode='constant', constant_values=0)
    padded_protein3 = np.pad(protein_3, ((0, padding_tensor3), (0, 0)), mode='constant', constant_values=0)

    protein_list1.append(padded_protein1)
    protein_list2.append(padded_protein2)
    protein_list3.append(padded_protein3)

    return protein_list1, protein_list2, protein_list3


if __name__ == '__main__':
    import random
    args = setup_args()
    model_path = args.model_path

    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    model = LaMGen_triple(pretrain_path='../Pretrained_model', config=Ada_config)

    param_dict = {key.replace("module.", ""): value for key, value in
                  torch.load(model_path, map_location='cuda').items()}

    model.load_state_dict(param_dict, strict=True)

    val_p1, val_p2, val_p3 = load_protein()

    all_output = []

    start_time = time.time()
    # Total number = range * batch size
    for protein1, protein2, protein3 in tqdm(zip(val_p1, val_p2, val_p3)):
        Seq_all = []
        for i in range(args.epochs):
            # random.seed(i)
            Seq_list = predict(model, tokenizer, protein1=protein1, protein2=protein2, protein3=protein3, seed=i, batch_size=args.batch_size)
            Seq_all.extend(Seq_list)
        for j in Seq_all:
            decoded_seq = decode(j)
            all_output.append([decoded_seq])

    time_elapsed = (time.time() - start_time)
    print(f'Time elapsed: {time_elapsed}s')

    output = pd.DataFrame(all_output)
    output.to_csv(args.output_path, index=False, header=False, mode='w')






