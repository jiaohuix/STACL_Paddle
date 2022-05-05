# only windows can use workers>1
workers=1
TEXT=dataset/bstc
python paddleseq_cli/preprocess.py \
        --source-lang zh --target-lang en \
        --srcdict $TEXT/vocab.zh --tgtdict  $TEXT/vocab.en \
        --trainpref $TEXT/asr.bpe --validpref $TEXT/dev.bpe --testpref $TEXT/dev.bpe  \
        --destdir data_bin/bstc_bin --thresholdtgt 0 --thresholdsrc 0 \
        --workers $workers


''''''

test.zh-en.en.idx
preprocess.log
test.zh-en.en.bin
test.zh-en.zh.bin
train.zh-en.en.bin
test.zh-en.zh.idx
train.zh-en.zh.bin
train.zh-en.en.idx
train.zh-en.zh.idx
valid.zh-en.en.bin
valid.zh-en.en.idx
valid.zh-en.zh.bin
valid.zh-en.zh.idx