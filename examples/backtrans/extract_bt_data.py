# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# -*- coding: utf-8 -*-

import argparse
import fileinput

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract back-translations from the stdout of fairseq-generate. "
            "If there are multiply hypotheses for a source, we only keep the first one. "
        )
    )
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument(
        "--srclang", required=True, help="source language (extracted from H-* lines)"
    )
    parser.add_argument(
        "--tgtlang", required=True, help="target language (extracted from S-* lines)"
    )
    parser.add_argument("--minlen", type=int, help="min length filter")
    parser.add_argument("--maxlen", type=int, help="max length filter")
    parser.add_argument("--ratio", type=float, help="ratio filter")
    parser.add_argument("--min-lprob", type=float,default=None, help="min log probs")
    parser.add_argument("files", nargs="*", help="input files")
    args = parser.parse_args()

    def validate(src, tgt, lprob):
        srclen = len(src.split(" ")) if src != "" else 0
        tgtlen = len(tgt.split(" ")) if tgt != "" else 0
        if (
            (args.minlen is not None and (srclen < args.minlen or tgtlen < args.minlen))
            or (
                args.maxlen is not None
                and (srclen > args.maxlen or tgtlen > args.maxlen)
            )
            or (
                args.ratio is not None
                and (max(srclen, tgtlen) / float(min(srclen, tgtlen)) > args.ratio)
            )
            or (
                args.min_lprob is not None
                and lprob<args.min_lprob
            )
        ):
            return False
        return True

    def safe_index(toks, index, default):
        try:
            return toks[index]
        except IndexError:
            return default

    with open(args.output + "." + args.srclang, "w" ,encoding="utf-8") as src_h, open(args.output + "." + args.tgtlang, "w" ,encoding="utf-8") as tgt_h:
        raw_lines,out_lines=0,0
        for line in tqdm(fileinput.input(args.files, openhook=fileinput.hook_encoded("utf-8"))):
            if line.startswith("S-"):
                tgt = safe_index(line.rstrip().split("\t"), 1, "")
            elif line.startswith("H-"):
                if tgt is not None:
                    raw_lines+=1
                    src = safe_index(line.rstrip().split("\t"), 2, "")
                    lprob = float(safe_index(line.rstrip().split("\t"), 1, ""))
                    if validate(src, tgt, lprob):
                        out_lines+=1
                        print(src, file=src_h)
                        print(tgt, file=tgt_h)
                    tgt = None
        dropped_lines=raw_lines-out_lines
        drop_rate=(dropped_lines/raw_lines)*100
        print(f"Raw lines:{raw_lines}, Out lines: {out_lines}, {raw_lines-out_lines} [{drop_rate:.4f}%] lines dropped for len filt or lprob < {args.min_lprob}.")
"""
# 注意：src zh,tgt en是前向，用回译的语料en-zh抽取的才是 src=zh tgt=en(回译的原文是英，但是写tgt=en)
python scripts/bt_demo/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 1.5 --output output/ft_data --srclang en --tgtlang zh  --min-lprob -3 output/generate*.txt 
python scripts/bt_demo/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 1.5 --output output/ft_data --srclang en --tgtlang zh  --min-lprob -3 output/ft/generate*.txt
6867774it [00:33, 203617.98it/s]
Raw lines:2289258, Out lines: 2019684, 269574 [11.7756%] lines dropped for len filt or lprob < -3.0.

# ratio 2.5
$ python scripts/bt_demo/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 2.5 --output output/ft_data_r2.5 --srclang en --tgtlang zh  --min-lprob -3 output/ft/generate*.txt
6867774it [00:35, 195373.17it/s]
Raw lines:2289258, Out lines: 2282085, 7173 [0.3133%] lines dropped for len filt or lprob < -3.0.


# 0 2 3 (无1) 
$ python scripts/bt_demo/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 2.5 --output output/ft_data_r2.5new --srclang en --tgtlang zh  --min-lprob -3 output/ft/generate*.txt
6039870it [00:30, 196468.52it/s]
Raw lines:2013290, Out lines: 2007039, 6251 [0.3105%] lines dropped for len filt or lprob < -3.0.


python scripts/bt_demo/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 2.5 --output output/ccmt_ft --srclang en --tgtlang zh  --min-lprob -3 output/bt_ft_ccmt/ft/generate_zhen.txt
python scripts/bt_demo/extract_bt_data.py --minlen 1 --maxlen 250 --ratio 2.5 --output output/ccmt_bt --srclang zh --tgtlang en  --min-lprob -3 output/bt_ft_ccmt/bt/generate_enzh.txt


"""
if __name__ == "__main__":
    main()
