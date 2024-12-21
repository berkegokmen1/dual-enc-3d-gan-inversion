CUDA_VISIBLE_DEVICES=1 python3 /home/bilecen/berke/PanoHead/training/encoder/infer_new_video.py \
  --output_dir /home/bilecen/berke/PanoHead/comparison_new/psp \
  --checkpoint_path /home/bilecen/berke/PanoHead/old_all_results/synth05fromZero_bs6/iteration_200000.pt \
  --data_path /home/bilecen/3d_inversion/data/lpff_3DDFA/no_background_combined_processed \
  --synth_data_path /home/bilecen/3d_inversion/data/CelebA-HQ_1024-no-bg-processed \
  --num_samples 10000 \
  --generate_background 0 \
  --use_zplus 0 \
  --use_z 0 \
  --synth_data_ratio 1 \
  --only_use_first_stage 1 \
  --eval_mode \
  --use_visible_region_stitching 0 \
  --fid_trunc_psi 0.8 \
  --save_images 0 \
  --save_videos 1 \
  --fid_pose_cond all \
  --generate_geometry 0 \
  --geometry_output_ply 0 \
  --use_w_opt 0 \
  --use_gen_opt 0


  # 

  # --checkpoint_path /home/bilecen/berke/PanoHead/old_all_results/exp-fixed-discriminator-15k-occlusion-2d-only-psp-d-loss-adv-lambda-0_005/checkpoints/iteration_320000.pt \
  # --visible_region_net_checkpoint_path /home/bilecen/berke/PanoHead/final_ckpts/z_plus_iteration_280000.pt \