ngpus=1
python paddleseq_cli/train.py --cfg configs/zhen_ccmt.yaml \
                         --amp \
                         --ngpus $ngpus   \
                         --update-freq 4 \
                         --max-epoch 10 \
                         --save-epoch 1 \
                         --save-dir /root/paddlejob/workspace/output \
                         --log-steps 100 \
                         --max-tokens 4096