export CUDA_VISIBLE_DEVICES=''
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do python create_gameplay_data.py -s $i 2>/dev/null & done