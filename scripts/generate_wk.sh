k=5
stream_prefix=dataset/stream_zh/dev/3
ckpt_dir=model_best_zhen
python paddleseq_cli/generate.py --cfg configs/zhen_waitk.yaml \
            --test-pref $stream_prefix --only-src \
            --pretrained  $ckpt_dir \
            --waitk $k --stream \
            --infer-bsz 1 --beam-size 5