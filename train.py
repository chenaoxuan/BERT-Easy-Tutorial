import argparse

import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from bert.model import BERT, BERTLM
from bert.dataset import WordVocab, BERTDataset


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", type=str, default="data/corpus.txt", help="train dataset path")
    parser.add_argument("-v", "--vocab_path", type=str, default="data/vocab", help="vocab path after prepare_vocab.py")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of bert model")
    parser.add_argument("-l", "--layers", type=int, default=6, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of transformer heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=1, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of Adam")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of Adam")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len)

    print("Creating Dataloader")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
    model = BERTLM(bert, len(vocab))

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for cur_epoch in range(args.epochs):
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        for i, data in enumerate(train_loader):

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = model.forward(data["bert_input"], data["segment_label"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss

            # 3. backward and optimization only in train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": cur_epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": round(loss.item(), 4)
            }

            if i % args.log_freq == 0:
                print(str(post_fix))

        print("Epoch%d, avg_loss=" % cur_epoch, avg_loss / len(train_loader), "total_acc=",
              total_correct * 100.0 / total_element)
