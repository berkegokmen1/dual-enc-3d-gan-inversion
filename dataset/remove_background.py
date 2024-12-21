import os
import PIL.Image
import torch
import argparse
from carvekit.api.high import HiInterface
from tqdm import tqdm
import numpy as np
import cv2


def blur_background(img_no_bg, image):
    image = np.array(image)
    mask = np.array(img_no_bg)[:, :, 3]

    mask = mask / 255.0
    mask = mask[:, :, np.newaxis]

    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 10)

    blurred_background = cv2.GaussianBlur(image, (31, 31), 0)

    blurred_mask = blurred_mask[:, :, np.newaxis]

    final_image = (image * blurred_mask + blurred_background * (1 - blurred_mask)).astype(np.uint8)

    final_image = PIL.Image.fromarray(final_image)

    return final_image


def process_images_in_batches(interface, images, output_dir, img_names, ext, blur_background=False):
    images_without_background = interface(images)  # This function might need to be adjusted to return masks

    # Process each image
    for img_no_bg, img, name in zip(images_without_background, images, img_names):
        img_name = name.split(".")[0]
        img_save_name = img_name + ext

        # Blur the background
        img = blur_background(img_no_bg, img) if blur_background else img_no_bg
        img = img.convert("RGB")
        img = img.resize((512, 512))

        # Save the image
        img_path = os.path.join(output_dir, img_save_name)
        img.save(img_path)


if __name__ == "__main__":
    print("Starting...")

    parser = argparse.ArgumentParser(description="Remove background from images")
    parser.add_argument("--input_dir", type=str, help="Path to the input directory", required=True)
    parser.add_argument("--output_dir", type=str, help="Path to the output directory", required=True)
    parser.add_argument("--save_ext", type=str, help="File extension", required=True)
    parser.add_argument("--batch_size", type=int, default=10, help="Number of images to process in a batch")
    parser.add_argument("--num_images", type=int, default=20, help="Number of images to process in a batch")
    parser.add_argument("--blur_background", type=int, default=0, help="Whether to blur the background or not")

    args = parser.parse_args()

    if "png" not in args.save_ext and "jpg" not in args.save_ext:
        raise ValueError("Only PNG and JPG formats are supported")

    if "." not in args.save_ext:
        args.save_ext = "." + args.save_ext

    os.makedirs(args.output_dir, exist_ok=True)

    img_names = [fname for fname in os.listdir(args.input_dir) if fname.endswith(".png") or fname.endswith(".jpg")]
    total = len(img_names) #NOTE changed this: min(len(img_names), args.num_images)

    print("Loading background remover model...")
    interface = HiInterface(
        object_type="object",  # Can be "object" or "hairs-like".
        batch_size_seg=1,
        batch_size_matting=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
        matting_mask_size=1024,
        trimap_prob_threshold=231,  # 231
        trimap_dilation=15,
        trimap_erosion_iters=20,
        fp16=True,
    )

    print("Removing backgrounds...")

    # Process images in batches
    for i in tqdm(range(0, total, args.batch_size)):
        batch_names = img_names[i : i + args.batch_size]
        batch_images = [PIL.Image.open(os.path.join(args.input_dir, name)) for name in batch_names]
        process_images_in_batches(interface, batch_images, args.output_dir, batch_names, args.save_ext)
