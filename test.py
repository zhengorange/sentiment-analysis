import argparse
import torch
from transformers import BertTokenizer
from utils import test, write_log
from model import TextClassification
from utils import MyDataLoader, MyDataSet

if __name__ == '__main__':
    weight_path = "weights/model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='测试参数设定')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument("--checkpoint", type=str,
                        default='hfl/chinese-macbert-base',
                        help='主干网络检查点：hfl/chinese-macbert-base，bert-base-chinese， hfl/chinese-roberta-wwm-ext')
    parser.add_argument("--device", type=str, help="选择GPU编号")
    args = parser.parse_args()
    if args.device is not None:
        device = torch.device(args.device)
    batch_size = args.batch_size
    checkpoint = args.checkpoint

    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    testDataLoader = MyDataLoader(batch_size=batch_size, dataset=MyDataSet(branch="test", tokenizer=tokenizer))
    model = TextClassification(checkpoint=checkpoint, freeze="0").to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device.type))
    test_acc = test(model=model,
                    data_loader=testDataLoader,
                    device=device,
                    tokenizer=tokenizer)

    write_log("test acc: %-10.3f " % test_acc, path="test_acc")

