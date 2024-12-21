CUDA_VISIBLE_DEVICES=0 python ./dataset/remove_background.py \
    --input_dir=<path-to-combined-dataset> \
    --output_dir=<output-path> \
    --save_ext=jpg \
    --batch_size=4 \
    --num_images=20 

