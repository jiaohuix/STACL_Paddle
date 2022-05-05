eval_folder=$1
trans_txt=$2 # A text file includes 500 lines of translation,
data_path=${eval_folder}/ref_for_mteval
script_path=${eval_folder}/eval_scripts
ref_sgm=${data_path}/dev16talks.ref.sgm
asr_sgm=${data_path}/dev16talks.src.asr.sgm
# each line is of concatenated translation of a talk.
# 16 is the number of talks in dev dataset.

if [ ! -d temp ]; then
    mkdir temp
fi

perl ${script_path}/bin/ref2frame.pl < $ref_sgm > temp/ref.frame.sgm
perl ${script_path}/bin/unique-blank.pl < $trans_txt | perl ${script_path}/bin/wrap-xml.perl temp/ref.frame.sgm eng USERNAME > temp/trans.sgm
perl ${script_path}/bin/mteval-v13a-1.pl -b -d 1 -s $asr_sgm -r $ref_sgm -t temp/trans.sgm

