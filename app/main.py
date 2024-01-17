import torch
import cv2
import numpy as np
import easyocr
import shutil
import math
import os
import argparse

best_trained_model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="model-weights/best.pt",
)


def preprocess_image(img):
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(converted, 11, 17, 17)
    inverted = cv2.bitwise_not(blur)
    threshold = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    return threshold


def get_region_of_interest(image):
    if image is None:
        print("Input Image is None!")
    else:
        print("Input Image is NOT None!")

    results = best_trained_model(image).xyxy[0]

    roi = None

    if results.any():
        bounded_box = results[0].cpu().numpy()

        xmin = int(bounded_box[0])
        xmax = int(bounded_box[2])
        ymin = math.ceil(bounded_box[1])
        ymax = math.ceil(bounded_box[3])

        roi = image[ymin:ymax, xmin:xmax]

    return roi


reader = easyocr.Reader(["en"], gpu=True)


def narrow_down_region_of_interest(roi):
    preprocessed_roi = preprocess_image(roi)

    ocr_results = reader.readtext(preprocessed_roi)

    narrowed_down_roi = None

    for ocr_result in ocr_results:
        image_area = roi.shape[0] * roi.shape[1]

        bounded_box_points = ocr_result[0]

        xmin = int(bounded_box_points[0][0])
        ymin = int(bounded_box_points[1][1])
        xmax = math.ceil(bounded_box_points[1][0])
        ymax = math.ceil(bounded_box_points[2][1])

        bounded_box_area = (xmax - xmin) * (ymax - ymin)

        if bounded_box_area / image_area >= 0.3:
            narrowed_down_roi = roi[ymin:ymax, xmin:xmax]

    return narrowed_down_roi


def contour_detection(narrowed_down_roi):
    preprocessed_nd_roi = preprocess_image(narrowed_down_roi)
    contours, _ = cv2.findContours(
        preprocessed_nd_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    filtered_contours = []

    for sorted_contour in sorted_contours:
        image_h, image_w = preprocessed_nd_roi.shape[0], preprocessed_nd_roi.shape[1]

        x, y, w, h = cv2.boundingRect(sorted_contour)

        if x - 3 < 0 or y - 3 < 0:  # disregard outer box contour
            continue

        if (
            h / image_h < 0.5 or h / image_h > 0.95
        ):  # disregard very small and very large contours
            continue

        filtered_contours.append(sorted_contour)

    return sorted(filtered_contours, key=lambda x: cv2.boundingRect(x)[0])


def extract_characters(narrowed_down_roi):
    contours = contour_detection(narrowed_down_roi)

    temp_dir = os.path.join("temp")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        character_image = narrowed_down_roi[
            y - 1 : y + h + 1, x - 1 : x + w + 1
        ]  # buffer space
        resized_character_image = cv2.resize(character_image, (28, 28))
        cv2.imwrite(f"temp/{index}.png", character_image)

    characters = []

    for c in os.listdir(temp_dir):
        cimg = cv2.imread(os.path.join(temp_dir, c))
        preprocessed_img = preprocess_image(cimg)
        results = reader.readtext(preprocessed_img)

        if results:
            characters.append(results[0][1])

    shutil.rmtree(temp_dir)

    return "".join(characters)


def clean_output(output_license_plate):
    alphanumeric_license_plate = "".join(
        [c for c in output_license_plate if c.isalnum()]
    )
    return alphanumeric_license_plate.upper()


def get_plate_number(input_image):
    roi = get_region_of_interest(input_image)

    if roi is not None:
        narrowed_down_roi = narrow_down_region_of_interest(roi)

        if narrowed_down_roi is not None:
            character_segmentation_plate_number = clean_output(
                extract_characters(narrowed_down_roi)
            )

            whole_image_results = reader.readtext(preprocess_image(narrowed_down_roi))

            whole_image_plate_number = ""

            if whole_image_results:
                whole_image_plate_number = clean_output(whole_image_results[0][1])

            return (
                character_segmentation_plate_number,
                whole_image_plate_number,
                narrowed_down_roi,
            )
        return "", "", None
    return "", "", None


parser = argparse.ArgumentParser(description="Process some images.")
parser.add_argument("image_path", type=str, help="Path to the image file")

args = parser.parse_args()
input_image = cv2.imread(args.image_path)

print("input image = ", input_image)

print(get_plate_number(input_image)[1])
