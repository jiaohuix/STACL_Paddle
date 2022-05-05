ref_sgm="500sents.ref.sgm"
transcript_sgm="500sents.src.transcript.sgm"
trans_txt="trans.txt" # A text file includes 500 lines of translation,
# each line record the concatenated translation of one source line.
# 500 is the number of sentences in dev dataset (before processed into streaming).

if [ ! -d temp ]; then
    mkdir temp
fi

perl bin/ref2frame.pl < $ref_sgm > temp/ref.frame.sgm
perl bin/unique-blank.pl < $trans_txt | perl bin/wrap-xml.perl temp/ref.frame.sgm spa USERNAME > temp/trans.sgm
perl bin/mteval-v13a-1.pl -b -d 1 -s $asr_sgm -r $ref_sgm -t temp/trans.sgm

