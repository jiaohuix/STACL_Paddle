if [ $# -lt 3 ];then
  echo "usage: bash $0 mode[dev/test](required) waitk(required) pretrained(required) beam_size(optional) workers(optional)"
  exit
fi
# params
mode=$1
k=$2
pretrained=$3
beam_size=1
workers=1
if [ $# -ge 4 ];then
  beam_size=$4
fi

if [ $# -ge 5 ];then
  workers=$5
fi
echo "-------------[mode: ${mode}, waitk: ${k}, pretrained: ${pretrained}, beam_size: ${beam_size}, workers: ${workers}]-------------"

src_lang=zh
tgt_lang=en
stream_folder=dataset/stream_zh/${mode}/
savedir=decode


if [ -e decode/zh-en.w${k}.all ]; then
    rm decode/zh-en.w${k}.all
fi
touch decode/zh-en.w${k}.all
if [ $mode == dev ];then
  file_nums=(3913 105 6634 2 111 4093 3075 2956 108 3 42 48 27 67 107 3063)
else
  file_nums=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
fi

file_len=${#file_nums[*]}
for i in $(seq 0 $workers $file_len);
do
  for ((j=i;j<i+$workers;j++))
    do
      (
      if [ $j -lt $file_len ];then
        prefix=${file_nums[$j]}
        echo "----------------------------generating file ${prefix}----------------------------"
        python paddleseq_cli/generate.py --cfg configs/zhen_ccmt.yaml \
                           --src-lang $src_lang  --test-pref $stream_folder/$prefix \
                           --pretrained $pretrained \
                           --waitk ${k} \
                           --stream \
                           --infer-bsz 1 \
                           --beam-size $beam_size \
                           --generate-path zh-en.w${k}.${prefix} \
                           --save-dir $savedir

        cat decode/zh-en.w${k}.${prefix} >> decode/zh-en.w${k}.all
        head -n 20 decode/zh-en.w${k}.${prefix}

      fi
      )&
    done
    wait
done

# evaluation for dev set
if [ $mode == dev ];then
  stream_out=decode/zh-en.w${k}.all
  jsonfile=decode/zh-en.dev.zh.json
  eval_folder=./dataset/Zh-En/dev/reference_eval
  python tools/latency.py $stream_out $jsonfile
  bash ${eval_folder}/eval_scripts/demo_mteval.sh  ${eval_folder}  ./decode/zh-en.w${k}.all.merge
fi

if [ $mode == test ];then
  folder=res_w${k}
  if [ ! -e  ${folder} ];then
    mkdir ${folder}
  fi

  for id in $(seq 1 20);do
    paste dataset/Zh-En/test/streaming_transcription/${id}.txt  decode/zh-en.w${k}.${id} > ${folder}/${id}.trans.txt
  done

  zip -r res_zh_en_T.zip res_w*

fi

echo "done"
