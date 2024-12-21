import json
import os

import numpy as np
from PIL import Image, ImageOps


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


if __name__ == "__main__":
    ffhq_dataset_dir = "/home/bilecen/3d_inversion/data/lpff/ffhq"
    ffhq_camera_params = "/home/bilecen/3d_inversion/data/lpff/ffhq/dataset.json"

    lpff_dataset_dir = "/home/bilecen/3d_inversion/data/lpff/eg3d"
    lpff_camera_params = "/home/bilecen/3d_inversion/data/lpff/camera_parameters.json"

    output_dir = "/home/bilecen/3d_inversion/data/lpff/combined_ffhq_lpff"

    os.makedirs(output_dir, exist_ok=True)

    with open(ffhq_camera_params, "r") as f:
        ffhq_dataset = json.load(f)

    with open(lpff_camera_params, "r") as f:
        lpff_dataset = json.load(f)

    new_dataset = ffhq_dataset.copy()

    index = int((new_dataset["labels"][-1][0]).replace("img", "").replace(".png", "")) + 1

    for file_name, camera_params in lpff_dataset.items():
        image_path = lpff_dataset_dir + "/" + file_name + ".png"

        # check if file in image_path exists
        try:
            with open(image_path):
                pass
        except FileNotFoundError:
            print("File not found:", image_path)
            continue

        print("Processing", image_path)

        img = Image.open(image_path)
        mirror_image = ImageOps.mirror(img)

        pose, intrinsics = np.array(camera_params[:16]).reshape(4, 4), np.array(camera_params[16:]).reshape(3, 3)
        flipped_pose = flip_yaw(pose)
        mirror_camera_params = np.concatenate([flipped_pose.reshape(-1), intrinsics.reshape(-1)]).tolist()

        new_dataset["labels"].append([f"img{index:08d}.png", camera_params])
        new_dataset["labels"].append([f"img{index + 1:08d}.png", mirror_camera_params])

        img.save(output_dir + f"/img{index:08d}.png")
        mirror_image.save(output_dir + f"/img{index + 1:08d}.png")

        print("Saved", output_dir + f"/img{index:08d}.png", output_dir + f"/img{index + 1:08d}.png")

        index += 2

    with open(output_dir + "/dataset.json", "w") as f:
        json.dump(new_dataset, f)

    print("Done")

    print(f"run cp -r {ffhq_dataset_dir}/* {output_dir}/")













