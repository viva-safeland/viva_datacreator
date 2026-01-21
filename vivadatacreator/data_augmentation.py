import os
import random
from PIL import Image
from tqdm import tqdm
import math
import argparse

def generate_dataset(root_path, output_path, final_shape, num_images, min_short_side_size, rotation_range=(0, 0)): 
    """ 
    Generates a dataset of image pairs (original and segmented). 
    """
    final_width, final_height = final_shape
    if min_short_side_size < min(final_width, final_height): 
        print(f"Warning: 'min_short_side_size' ({min_short_side_size}) is smaller than the smallest final dimension ({min(final_width, final_height)}).") 
        print(f"To prevent upscaling and quality loss, the minimum crop size will be forced to {min(final_width, final_height)}.") 
        min_short_side_size = min(final_width, final_height)

    output_imgs_path = os.path.join(output_path, 'imgs') 
    output_segs_path = os.path.join(output_path, 'segmented') 
    os.makedirs(output_imgs_path, exist_ok=True) 
    os.makedirs(output_segs_path, exist_ok=True) 
    
    valid_image_pairs = [] 
    try: 
        subfolders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))] 
        if not subfolders: 
            print("Error: No subfolders found in the root path.") 
            return 

        for folder in subfolders: 
            imgs_folder = os.path.join(root_path, folder, 'imgsA') 
            segs_folder = os.path.join(root_path, folder, 'dataset') 

            if not os.path.isdir(imgs_folder) or not os.path.isdir(segs_folder): 
                continue 

            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'} 
            for filename in os.listdir(imgs_folder): 
                if os.path.splitext(filename)[1].lower() in valid_extensions: 
                    base_name = os.path.splitext(filename.lstrip('0'))[0] 
                    seg_name = base_name + '.png' 
                    img_path = os.path.join(imgs_folder, filename) 
                    seg_path = os.path.join(segs_folder, seg_name) 

                    if os.path.isfile(seg_path): 
                        valid_image_pairs.append((img_path, seg_path)) 
        
        if not valid_image_pairs: 
            print("Error: No valid image/segmentation pairs were found in the specified structure.") 
            return 

    except FileNotFoundError: 
        print(f"Error: The root path '{root_path}' was not found.") 
        return 

    generated_count = 0 
    rotation_active = rotation_range[0] != 0 or rotation_range[1] != 0
    aspect_ratio = final_width / final_height

    for i in tqdm(range(num_images), desc="Generating Images"): 
        try: 
            img_path, seg_path = random.choice(valid_image_pairs) 

            with Image.open(img_path) as img, Image.open(seg_path) as seg_img: 
                original_width, original_height = img.size 
                
                max_possible_source_crop = min(original_width, original_height)
                
                # Based on D = h * sqrt(aspect_ratio^2 + 1)
                max_h = max_possible_source_crop / math.sqrt(aspect_ratio**2 + 1)
                max_w = max_h * aspect_ratio
                
                # Determine min possible inner crop size
                if aspect_ratio >= 1:
                    min_h = min_short_side_size / aspect_ratio
                    min_w = min_short_side_size
                else: # Portrait
                    min_h = min_short_side_size
                    min_w = min_short_side_size * aspect_ratio

                if min_h > max_h or min_w > max_w:
                    tqdm.write(f"Skipping {os.path.basename(img_path)}: too small for specified crop size.")
                    continue
                
                inner_crop_h = random.randint(int(min_h), int(max_h))
                inner_crop_w = int(inner_crop_h * aspect_ratio)

                source_crop_size = int(math.sqrt(inner_crop_w**2 + inner_crop_h**2)) if rotation_active else max(inner_crop_w, inner_crop_h)

                max_x = original_width - source_crop_size 
                max_y = original_height - source_crop_size 
                start_x = random.randint(0, max_x) 
                start_y = random.randint(0, max_y) 
                
                source_box = (start_x, start_y, start_x + source_crop_size, start_y + source_crop_size)
                source_img = img.crop(source_box)
                source_seg = seg_img.crop(source_box)
                
                random_angle = random.uniform(rotation_range[0], rotation_range[1])
                rotated_img = source_img.rotate(random_angle, resample=Image.Resampling.BICUBIC, expand=False)
                rotated_seg = source_seg.rotate(random_angle, resample=Image.Resampling.NEAREST, expand=False)
                
                center = source_crop_size / 2
                half_w, half_h = inner_crop_w / 2, inner_crop_h / 2
                inner_box = (center - half_w, center - half_h, center + half_w, center + half_h)
                
                final_crop_img = rotated_img.crop(inner_box)
                final_crop_seg = rotated_seg.crop(inner_box)
                
                final_img = final_crop_img.resize(final_shape, Image.Resampling.LANCZOS)
                final_seg = final_crop_seg.resize(final_shape, Image.Resampling.NEAREST)

                base_name = os.path.splitext(os.path.basename(img_path).lstrip('0'))[0] 
                output_img_name = f"{i}_{base_name}.jpg" 
                output_seg_name = f"{i}_{base_name}.png" 
                
                output_img_path = os.path.join(output_imgs_path, output_img_name) 
                output_seg_path = os.path.join(output_segs_path, output_seg_name) 

                final_img.save(output_img_path) 
                final_seg.save(output_seg_path) 
                generated_count += 1 

        except Exception as e: 
            tqdm.write(f"An error occurred during iteration {i + 1}: {e}. Continuing...") 

    print(f"\nProcess complete! Generated {generated_count} images.")

def main():
    parser = argparse.ArgumentParser(description="Generate augmented segmentation dataset.")
    parser.add_argument("--root", required=True, help="Root directory containing video folders")
    parser.add_argument("--output", required=True, help="Output directory for generated dataset")
    parser.add_argument("--width", type=int, default=1056, help="Final image width")
    parser.add_argument("--height", type=int, default=704, help="Final image height")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to generate")
    parser.add_argument("--min-size", type=int, default=300, help="Minimum short side size for cropping")
    parser.add_argument("--min-rot", type=float, default=0, help="Minimum rotation angle")
    parser.add_argument("--max-rot", type=float, default=360, help="Maximum rotation angle")

    args = parser.parse_args()

    generate_dataset(
        root_path=args.root,
        output_path=args.output,
        final_shape=(args.width, args.height),
        num_images=args.num_images,
        min_short_side_size=args.min_size,
        rotation_range=(args.min_rot, args.max_rot)
    )

if __name__ == "__main__":
    main()
