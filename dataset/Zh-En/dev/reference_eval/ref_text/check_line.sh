file_nums=(3913 105 6634 2 111 4093 3075 2956 108 3 42 48 27 67 107 3063)
folder=./
for idx in ${file_nums[*]};do
  lines=$(cat $folder/${idx}.txt | wc -l)
  echo "file ${idx}.txt lines: ${lines}"
done