# params
if [ $# -lt 3 ];then
  echo "usage: bash $0 mode[dev/test](required) waitk(required) pretrained(required) beam_size(optional)"
  exit
fi

mode=$1 # dev or test
k=$2
pretrained=$3
beam_size=1
if [ $# -ge 4 ];then
  beam_size=$4
fi

src_lang=en
tgt_lang=es
stream_prefix=dataset/stream_en/${mode}.bpe
savedir=output

# waitk generate
python paddleseq_cli/generate.py --cfg configs/enes_un.yaml \
                   --src-lang $src_lang  --test-pref $stream_prefix \
                   --pretrained $pretrained \
                   --waitk ${k} \
                   --stream \
                   --infer-bsz 1 \
                   --beam-size $beam_size \
                   --generate-path ${src_lang}-${tgt_lang}.w${k}.all \
                   --save-dir $savedir \


cp $savedir/${src_lang}-${tgt_lang}.w${k}.all decode/${src_lang}-${tgt_lang}.w${k}.all

# evaluation for dev set
if [ $mode == dev ];then
  stream_out=decode/en-es.w${k}.all
  jsonfile=decode/en-es.dev.en.json
  eval_folder=dataset/En-Es/dev/reference_eval
  python tools/latency.py $stream_out $jsonfile
  bash ${eval_folder}/eval_scripts/demo_mteval.sh  ${eval_folder}  ./decode/en-es.w${k}.all.merge
fi


if [ $mode == test ];then
  if [ ! -d res_w${k} ];then
    mkdir res_w${k}
  fi
  paste dataset/En-Es/test/streaming_transcription/en-es.stream.test.src.txt  $savedir/en-es.w${k}.all > res_w${k}/res.trans.txt

  zip -r res_en_es_T.zip res_w*

fi

echo "done"
