# only windows can use workers>1
workers=1
TEXT=dataset/bstc
python paddleseq_cli/preprocess.py \
        --source-lang zh --target-lang en \
        --srcdict $TEXT/vocab.zh --tgtdict  $TEXT/vocab.en \
        --trainpref $TEXT/asr.bpe --validpref $TEXT/dev.bpe --testpref $TEXT/dev.bpe  \
        --destdir data_bin/bstc_bin --thresholdtgt 0 --thresholdsrc 0 \
        --workers $workers
