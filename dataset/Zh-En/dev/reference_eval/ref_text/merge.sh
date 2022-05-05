file_nums=(3913 105 6634 2 111 4093 3075 2956 108 3 42 48 27 67 107 3063)

touch dev.en

for idx in ${file_nums[*]};do
  echo $idx
  # merge
  cat ${idx}.txt >> dev.en
done

echo "done!"