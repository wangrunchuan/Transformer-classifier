import argparse


class Hparams:
    parser = argparse.ArgumentParser()

    parser.add_argument('--en_vocab_size', default=24000, type=int)
    parser.add_argument('--de_vocab_size', default=24000, type=int)

    # parser.add_argument('--train_fp', default='./data/imdb/train_data.txt')
    # parser.add_argument('--eval_fp', default='./data/imdb/eval_data.txt')
    parser.add_argument('--en_fp', default='./data/iwslt/train.en')
    parser.add_argument('--de_fp', default='./data/iwslt/train.de')
    parser.add_argument('--en_pretreat_path', default='./data/iwslt/pretreat/en')
    parser.add_argument('--de_pretreat_path', default='./data/iwslt/pretreat/de')
    parser.add_argument('--ckpt_path', default='./data/iwslt/ckpt')
    parser.add_argument('--stop_word_path', default=None)
    parser.add_argument('--en_w2v_model_path', default='./data/iwslt/pretreat/en/w2v_model.en')
    parser.add_argument('--de_w2v_model_path', default='./data/iwslt/pretreat/de/w2v_model.de')

    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--sequence_length', default=512, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--l2RegLambda', default=0.0, type=float)

    parser.add_argument('--d_model', default=256, type=int,
                        help="hidden dimension of encoder/decoder")  # 512
    parser.add_argument('--d_ff', default=128, type=int,
                        help="hidden dimension of feedforward layer")  # 2048
    parser.add_argument('--num_blocks', default=1, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--dropout_rate', default=0.3, type=float)

    parser.add_argument('--num_class', default=2, type=int)
