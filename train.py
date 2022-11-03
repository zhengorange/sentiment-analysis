import argparse
import os
import torch
from transformers import BertTokenizer
from model import TextClassification
from utils import train_one_epoch, evaluate, MyDataLoader, MyDataSet, write_log
import tensorboardX

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设定')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--lr', type=float, default=2e-5, help='lr')
    parser.add_argument("--checkpoint", type=str,
                        default='hfl/chinese-macbert-base',
                        help='主干网络检查点：hfl/chinese-macbert-base，bert-base-chinese， hfl/chinese-roberta-wwm-ext')
    parser.add_argument("--freeze", type=str,
                        default="0",
                        help='冻结主干网络模式。0：不冻结；1：冻结word embedding；2：冻结全部embeddings；3：encoder只解冻pooler')
    parser.add_argument("--device", type=str, help="选择GPU编号")
    args = parser.parse_args()

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    tb_writer = tensorboardX.SummaryWriter("log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device is not None:
        device = torch.device(args.device)
    epoch_num = args.epoch
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    freeze = args.freeze
    lr = args.lr

    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    trainDataLoader = MyDataLoader(batch_size=batch_size, dataset=MyDataSet(branch="train", tokenizer=tokenizer))
    verifyDataLoader = MyDataLoader(batch_size=batch_size, dataset=MyDataSet(branch="verify", tokenizer=tokenizer))

    model = TextClassification(checkpoint=checkpoint, freeze=freeze).to(device)

    parameters_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters_update, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    verify_acc = 0
    for epoch in range(epoch_num):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=trainDataLoader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=verifyDataLoader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if verify_acc < val_acc:
            verify_acc = val_acc
            torch.save(model.state_dict(), "./weights/model.pth".format(epoch, verify_acc))
            write_log("epoch: {}, verify_acc: {}".format(epoch, verify_acc), path="./model_save.txt")
