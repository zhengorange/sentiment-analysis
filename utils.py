import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.BCEWithLogitsLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids, attention_mask, token_type_ids, labels = (x.to(device) for x in data)
        sample_num += input_ids.shape[0]
        out = model(enc_inputs=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        out = out.view(-1)
        predict_classes = torch.round(torch.sigmoid(out))
        accu_num += torch.eq(predict_classes, labels).sum()
        loss = loss_function(out, labels.float())
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.BCEWithLogitsLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids, attention_mask, token_type_ids, labels = (x.to(device) for x in data)
        sample_num += input_ids.shape[0]
        out = model(enc_inputs=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        out = out.view(-1)
        predict_classes = torch.round(torch.sigmoid(out))
        accu_num += torch.eq(predict_classes, labels).sum()
        loss = loss_function(out, labels.float())
        accu_loss += loss.detach()
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def test(model, data_loader, device, tokenizer):
    write_log("true", "prec", "content", path="./test_err_log.txt")
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids, attention_mask, token_type_ids, labels = (x.to(device) for x in data)
        sample_num += input_ids.shape[0]
        out = model(enc_inputs=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        out = out.view(-1)
        predict_classes = torch.round(torch.sigmoid(out))
        accu_num += torch.eq(predict_classes, labels).sum()
        contents = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        for i in range(input_ids.shape[0]):
            if labels[i] != predict_classes[i]:
                write_log(labels[i], predict_classes[i], contents[i], path="./test_err_log.txt")
        write_log("batch acc: ", torch.eq(predict_classes, labels).sum() / input_ids.shape[0], path="./test_err_log.txt")
        data_loader.desc = "[test] acc: {:.3f}".format(accu_num.item() / sample_num)
    return accu_num.item() / sample_num


class MyDataSet(Dataset):
    def __init__(self, branch='train', tokenizer=None):
        super(MyDataSet, self).__init__()
        df = pd.read_csv("dataset/%s.tsv" % branch, encoding='utf-8', sep="\t").fillna('')
        self.tokenizer = tokenizer
        self.data = df.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc_input = self.tokenizer(
            self.data[idx][1],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc_input.input_ids.squeeze(), enc_input.attention_mask.squeeze(), enc_input.token_type_ids.squeeze(), \
               self.data[idx][0]


def MyDataLoader(batch_size: int, dataset=None):
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    return data_loader


def write_log(*t, path="./log.txt"):
    t = ",".join([str(item) for item in t])
    f = open(path, "a")
    f.write(t + '\n')
    f.close()
