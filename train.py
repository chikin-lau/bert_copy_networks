import argparse
from Trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_dir', default='./data/CPC',
                        help='directory for dataset')
    parser.add_argument('--train_file', default='Persona_train_1W.tsv',
                        help='name of train file')
    parser.add_argument('--dev_file', default='Persona_val_sample.tsv',
                        help='name of dev file')
    parser.add_argument('--test_file', default='Persona_test_deepclean.tsv',
                        help='name of test file')
    parser.add_argument('--vocab_path', default='./model_dict/vocab.txt',
                        help='vocab path for pre-trained model')
    parser.add_argument('--max_len', type=int, default=64,
                        help='max length of source data')
    parser.add_argument('--tgt_len', type=int, default=64,
                        help='max length of target data')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=32,
                        help='batch size for validation')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='batch size for evaluating')
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of epochs for training')
    parser.add_argument('--label_smooth', default=0.1, type=float,
                        help='label smoothing coeff')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus to use')
    parser.add_argument('--fp16', default=False, type=bool,
                        help='是否使用混合精度加速')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='learning rate')
    parser.add_argument('--pre_lr', default=7e-6, type=float,
                        help='pretrain model learning rate')
    parser.add_argument('--beam_size', default=3, type=int,
                        help='beam size')
    parser.add_argument('--top_k', default=5, type=int,
                        help='top_k size')
    parser.add_argument('--top_p', default=0.9, type=float,
                        help='top_p size')
    # parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--is_schedule', action='store_true', help='是否使用schedule')
    parser.add_argument('--is_b2b', action='store_true')
    parser.add_argument('--is_test', action='store_true')
    parser.add_argument('--is_generate', action='store_true')
    parser.add_argument('--is_eval', action='store_true')
    parser.add_argument('--is_beam_search', action='store_true')
    args = parser.parse_args()

    trainer = Trainer(args, rank=1)

    if args.is_test:
        trainer.test()
    elif args.is_generate:
        trainer.generate(out_max_length=64, top_k=args.top_k, top_p=args.top_p, max_length=128)
    elif args.is_eval:
        trainer.eval()
    elif args.is_beam_search:
        trainer.beam_search(beam_size=args.beam_size)
    else:
        trainer.train()
    # trainer.train()

    # trainer.test()
    # trainer.generate(out_max_length=60, top_k=5, top_p=0.95, max_length=200)
    # trainer.eval()


if __name__ == "__main__":
    main()




# !python train.py --epochs 15 --train_batch_size 16 --dev_batch_size 16 --is_schedule --train_file "Persona_train_clean.tsv" --dev_file "Persona_val_clean.tsv"
# !python train.py --is_eval --eval_batch_size 16 --test_file "Persona_test_deepclean.tsv"
# !python train.py --is_test --eval_batch_size 16 --test_file "Persona_test_deepclean.tsv"
# !python train.py --is_generate --top_k 5 --top_p 0.9 --eval_batch_size 16 --test_file "Persona_test_deepclean.tsv"
# !python train.py --is_beam_search --beam_size 3 --eval_batch_size 16 --test_file "Persona_test_deepclean.tsv"
# !python test.py --epochs 2

# --dataset_dir './data/ConvAI2'

# !python train.py --epochs 5 --tgt_len 32 --train_batch_size 16 --dev_batch_size 16 --is_schedule --dataset_dir './data/ConvAI2' --train_file "Persona_train_1W.tsv" --dev_file "Persona_val_1K.tsv" --test_file "Persona_test_4K.tsv"
