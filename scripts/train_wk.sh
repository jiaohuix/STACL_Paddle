k=5
python paddleseq_cli/train.py --cfg configs/zhen_waitk.yaml \
            --waitk $k --pretrained ckpt/model_best_zhen