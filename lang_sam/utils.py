import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import warnings
import requests
from lang_sam import LangSAM

MIN_AREA = 100


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def get_contours(mask):
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, 0)
    mask = mask.astype(np.uint8)
    mask *= 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    effContours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            effContours.append(c)
    return effContours


def contour_to_points(contour):
    pointsNum = len(contour)
    contour = contour.reshape(pointsNum, -1).astype(np.float32)
    points = [point.tolist() for point in contour]
    return points


def generate_labelme_json(binary_masks, labels, image_size, image_path=None):
    """Generate a LabelMe format JSON file from binary mask tensor.

    Args:
        binary_masks: Binary mask tensor of shape [N, H, W].
        labels: List of labels for each mask.
        image_size: Tuple of (height, width) for the image size.
        image_path: Path to the image file (optional).

    Returns:
        A dictionary representing the LabelMe JSON file.
    """
    num_masks = binary_masks.shape[0]
    binary_masks = binary_masks.numpy()

    json_dict = {
        "version": "4.5.6",
        "imageHeight": image_size[0],
        "imageWidth": image_size[1],
        "imagePath": image_path,
        "flags": {},
        "shapes": [],
        "imageData": None
    }

    # Loop through the masks and add them to the JSON dictionary
    for i in range(num_masks):
        mask = binary_masks[i]
        label = labels[i]
        effContours = get_contours(mask)

        for effContour in effContours:
            points = contour_to_points(effContour)
            shape_dict = {
                "label": label,
                "line_color": None,
                "fill_color": None,
                "points": points,
                "shape_type": "polygon"
            }

            json_dict["shapes"].append(shape_dict)

    return json_dict

def display_coins_with_boxes(image, boxes, labels, classes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis('off')

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        #label = round(label.item(), 2)  # Convert logit to a scalar before rounding
        coin_type = classes[label]
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(x_min, y_min, f"{coin_type}", fontsize=8, color='red', verticalalignment='top')

    plt.show()

def find_coin_masks(image):
    # Suppress warning messages
    warnings.filterwarnings("ignore")
    text_prompt = "coin"
    try:
        model = LangSAM()
        masks, boxes, phrases, logits = model.predict(image, text_prompt)

        if len(masks) == 0:
            print(f"No objects of the '{text_prompt}' prompt detected in the image.")
        else:
            # Convert masks to numpy arrays
            masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
            boxes_np = [box.squeeze().cpu().numpy() for box in boxes]
            return masks_np, boxes_np

    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Error: {e}")

    
# returns y_min, y_max, x_min, x_max
def find_boundary_of_coin(nonzero_indices):
    return nonzero_indices[0,:].min(), nonzero_indices[0,:].max(), nonzero_indices[1,:].min(), nonzero_indices[1,:].max()



def generate_coin_images(image_dir):
    image = Image.open(image_dir).convert("RGB")
    masks, boxes = find_coin_masks(image)
    image = np.array(image)
    coins = []
    for index in range(len(masks)):
        mask = np.broadcast_to(np.expand_dims(masks[index],-1), image.shape)
        masked_image = mask * image
        nonzero_indices = np.nonzero(masked_image[:,:,0])
        nonzero_indices = np.array(nonzero_indices)
        y_min, y_max, x_min, x_max = find_boundary_of_coin(nonzero_indices)
        masked_image = masked_image[y_min:y_max,x_min:x_max]
        if (y_max - y_min)<500 and (x_max - x_min)<500:
            difference_y = 500 - (y_max - y_min)
            difference_x = 500 - (x_max - x_min)
            if difference_y != 0:
                if difference_y % 2 == 0:
                    masked_image = np.pad(masked_image, [(difference_y//2, difference_y//2), (0, 0), (0, 0)])
                else:
                    masked_image = np.pad(masked_image, [((difference_y-1)//2, (difference_y-1)//2 + 1), (0, 0), (0, 0)])
            if difference_x != 0:
                if difference_x % 2 == 0:
                    masked_image = np.pad(masked_image, [(0, 0), (difference_x//2, difference_x//2), (0, 0)])
                else:
                    masked_image = np.pad(masked_image, [(0, 0), ((difference_x-1)//2, (difference_x-1)//2 + 1), (0, 0)])
            coins.append(masked_image)
    return coins, boxes