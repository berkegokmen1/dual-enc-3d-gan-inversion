dataset_paths = {
    "test": "<path-to-data>/no_background_combined_processed_test",
    "train": "<path-to-data>/no_background_combined_processed",
    "synth": "<path-to-data>/synth_z_plus_200k_trunc_08_no_bg",
}

model_paths = {
    "pano_head": "<path-to-ckpts>/ckpts/panohead/easy-khair-180-gpc0.8-trans10-025000.pkl",
    "latent_avg": "<path-to-ckpts>/ckpts/panohead/latent_avg.npy",
    "ir_se50": "<path-to-ckpts>/ckpts/panohead/model_ir_se50.pth",
    "circular_face": "<path-to-ckpts>/ckpts/panohead/CurricularFace_Backbone.pth",
    "bisenet": "<path-to-ckpts>/ckpts/bisenet/79999_iter.pth",
}
