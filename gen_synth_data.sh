
CUDA_VISIBLE_DEVICES=0 python3 dataset/gen_synth_data_z_plus.py \
    --output_dir <output-path> \
    --num_samples 200000 \
    --network <panohead-ckpt-path> \
    --bisenet_ckpt <bisenet-ckpt-path> \
    --use_mask_net \
    --trunc_psi 0.80 \
    --reload_generator \
