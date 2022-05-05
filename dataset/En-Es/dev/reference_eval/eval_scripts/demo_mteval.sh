eval_folder=$1
trans_txt=$2 # A text file includes 500 lines of translation,
tgt=spa # target lang id
data_path=${eval_folder}/ref_for_mteval
script_path=${eval_folder}/eval_scripts

ref_sgm=${data_path}/500sents.ref.sgm
transcript_sgm=${data_path}/500sents.src.transcript.sgm
# each line record the concatenated translation of one source line.
# 500 is the number of sentences in dev dataset (before processed into streaming).

if [ ! -d temp ]; then
    mkdir temp
fi

perl ${script_path}/bin/ref2frame.pl < $ref_sgm > temp/ref.frame.sgm # 将sgm的参考文本提到temp
perl ${script_path}/bin/unique-blank.pl < $trans_txt | perl ${script_path}/bin/wrap-xml.perl temp/ref.frame.sgm $tgt USERNAME > temp/trans.sgm # 按格式将预测结果提到temp的sgm
perl ${script_path}/bin/mteval-v13a-1.pl -b -d 1 -s $transcript_sgm -r $ref_sgm -t temp/trans.sgm # 根据-s原文 -r 译文 -t 翻译结果计算bleu得分

# -s source -r reference -t predict target