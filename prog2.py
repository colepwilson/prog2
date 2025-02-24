import cv2
import numpy as np
import sys

def compute_disparity(left_img_path, right_img_path, scale_factor, output_path):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if left_img is None or right_img is None:
        print("Error loading images")
        return

    height, width = left_img.shape
    disparity_map = np.zeros((height, width), dtype=np.uint8)

    max_disparity = 50

    block_size = 3
    half_block = block_size // 2

    for y in range(half_block, height - half_block):
        for x in range(half_block, width - half_block):
            min_ssd = float('inf')
            best_disparity = 0

            for d in range(max_disparity):
                if x - d < half_block:
                    continue

                left_patch = left_img[y-half_block:y+half_block+1, x-half_block:x+half_block+1]
                right_patch = right_img[y-half_block:y+half_block+1, x-half_block-d:x+half_block+1-d]

                if right_patch.shape != left_patch.shape:
                    continue

                ssd = np.sum((left_patch - right_patch) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disparity = d

            disparity_map[y, x] = int(best_disparity * scale_factor)

    cv2.imwrite(output_path, disparity_map)
    print(f"Disparity map saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prog2.py <left_image> <right_image> <scale_factor> <output_image>")
        sys.exit(1)

    left_image_path = sys.argv[1]
    right_image_path = sys.argv[2]
    scale_factor = float(sys.argv[3])
    output_image_path = sys.argv[4]

    compute_disparity(left_image_path, right_image_path, scale_factor, output_image_path)
