import argparse
from bert.dataset.vocab import WordVocab


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", type=str, default='data/corpus.txt')
    parser.add_argument("-o", "--output_path", type=str, default='data/vocab')
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)
