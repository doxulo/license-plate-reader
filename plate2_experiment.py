import cv2
import numpy as np

def box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    a_right = ax + aw
    a_bottom = ay + ah

    b_right = bx + bw
    b_bottom = by + bh

    # Compute intersection 
    intersection_left = max (ax, bx)
    intersection_top = max (ay, by)
    intersection_right = min(a_right, b_right)
    intersection_bottom = min(a_bottom, b_bottom)
    intersection_height = max(
        0,
        intersection_bottom - intersection_top
    )
    intersection_width = max(
        0,
        intersection_right - intersection_left
    )

    # Calculate union area
    area_a = aw * ah
    area_b = bw * bh
    intersection_area = intersection_width * intersection_height
    union_area = area_a + area_b - intersection_area

    # Calculate Intersection over Union
    iou = intersection_area / union_area

    return iou

def predict_character(target_character, templates):
    best_label = None
    best_score = -1

    for label, template_list in templates.items():
        if not template_list:
            continue
        
        label_scores = []

        for template in template_list:
            #test
            if template is target_character:
                continue
            #end test
            result = cv2.matchTemplate(
                target_character,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            score = result[0, 0]
            label_scores.append(score)
#test
        if not label_scores:
            continue
        #end
        average_score = sum(label_scores) / len(label_scores)
        if average_score > best_score:
            best_score = average_score
            best_label = label

    return best_label, best_score
# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_PATH = "plate2.jpg"

# Crop coordinates in the original image.
CROP_Y_START = 275
CROP_Y_END = 950
CROP_X_START = 325
CROP_X_END = 1250

# Canny edge-detection thresholds.
CANNY_LOW = 100
CANNY_HIGH = 200

# Character contour filters.
MIN_CONTOUR_AREA = 10
MIN_CHARACTER_HEIGHT = 180
MAX_CHARACTER_HEIGHT = 300
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 0.6

# Normalized dimensions
NORMALIZED_HEIGHT = 100
NORMALIZED_WIDTH = 60

# --------------------------------------------------
# Load the original image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)  

if image is None:
    print(f"Could not read image: {IMAGE_PATH}")

else:
    print(image.shape)
    display_scale = 0.4
    display_image = cv2.resize(
        image,
        None,
        fx=display_scale,
        fy=display_scale
    )
    x_small, y_small, width_small, height_small = cv2.selectROI(
        "Select License Plate",
        display_image,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyWindow("Select License Plate")

    x = round(x_small / display_scale)
    y = round(y_small / display_scale)
    width = round(width_small / display_scale)
    height = round(height_small / display_scale)

    image_height, image_width, image_channels = image.shape

    print("Original image:")
    print(f"  Height: {image_height}")
    print(f"  Width: {image_width}")
    print(f"  Channels: {image_channels}")

    # --------------------------------------------------
    # Crop the license plate
    # --------------------------------------------------

    # NumPy/OpenCV slicing uses:
    # image[y_start:y_end, x_start:x_end]
    plate_crop = image[
        y:y+height,
        x:x+width
    ]

    print("\nCropped plate shape:", plate_crop.shape)

    # --------------------------------------------------
    # Convert the plate to grayscale
    # --------------------------------------------------

    gray_plate = cv2.cvtColor(
        plate_crop,
        cv2.COLOR_BGR2GRAY
    )

    print("Grayscale plate shape:", gray_plate.shape)

    # --------------------------------------------------
    # Detect edges
    # --------------------------------------------------

    edges = cv2.Canny(
        gray_plate,
        CANNY_LOW,
        CANNY_HIGH
    )

    print("Edge image shape:", edges.shape)

    # --------------------------------------------------
    # Close small gaps in the detected edges
    # --------------------------------------------------

    # A 3x3 kernel examines the current pixel and
    # the eight pixels surrounding it.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    # Morphological closing performs:
    # dilation followed by erosion.
    #
    # It helps connect small gaps while keeping the
    # edges closer to their original thickness.
    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------
    # Find contours
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print("\nTotal contours found:", len(contours))

    # --------------------------------------------------
    # Filter contours that resemble characters
    # --------------------------------------------------

    character_contours = []
    character_boxes = []
    
    for contour in contours:
        contour_area = cv2.contourArea(contour)

        # Ignore extremely small contours first.
        if contour_area <= MIN_CONTOUR_AREA:
            continue

        # Find the upright bounding rectangle around
        # the current contour.
        x, y, box_width, box_height = cv2.boundingRect(contour)

        # Reject contours that are too short or too tall.
        if not MIN_CHARACTER_HEIGHT < box_height < MAX_CHARACTER_HEIGHT:
            continue

        # Aspect ratio tells us how wide the contour is
        # compared with its height.
        aspect_ratio = box_width / box_height

        # Reject contours that are too thin or too wide.
        if not MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO:
            continue

        # This contour passed every filter.
        character_contours.append(contour)
        character_boxes.append(
            (x, y, box_width, box_height)
        )
        print(
            f"Accepted contour:"
            f" x={x},"
            f" y={y},"
            f" width={box_width},"
            f" height={box_height},"
            f" area={contour_area:.1f},"
            f" ratio={aspect_ratio:.2f}"
        )

    print(
        "\nAccepted character contours:",
        len(character_contours)
    )

    character_boxes.sort(
        key = lambda box: box[2] * box[3],
        reverse = True
    )

    unique_boxes = []

    for candidate in character_boxes:
        is_duplicate = any(
            box_iou(candidate, kept_box) > 0.8
            for kept_box in unique_boxes
        )

        if not is_duplicate:
            unique_boxes.append(candidate)

    unique_boxes.sort(
        key = lambda box: (box[1], box[0]),
    )

    print("Unique boxes:")
    for box in unique_boxes:
        print(box)

    # Seperate boxes on to 2 top and bottom rows
    plate_middle_y = plate_crop.shape[0] / 2

    top_row = []
    bottom_row = []

    for box in unique_boxes:
        x, y, box_width, box_height = box
        center_y = y + box_height/2
        if center_y < plate_middle_y:
            top_row.append(box)
        else:
            bottom_row.append(box)

    top_row.sort(key=lambda box: box[0])
    bottom_row.sort(key=lambda box: box[0])

    print("Top row:")
    for box in top_row:
        print(box)

    print("Bottom row:")
    for box in bottom_row:
        print(box)

    ordered_boxes = top_row + bottom_row
    character_binaries = []
    for index, box in enumerate(ordered_boxes):
        x, y, box_width, box_height = box
        character_image = plate_crop[
            y: y + box_height,
            x: x + box_width 
        ]
        character_gray = cv2.cvtColor(
            character_image,
            cv2.COLOR_BGR2GRAY
        )
        threshold_used, character_binary = cv2.threshold(
            character_gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        character_binaries.append(character_binary)

    resized_characters = []
    for index, character_binary in enumerate(character_binaries):
        original_height, original_width = character_binary.shape

        height_scale = NORMALIZED_HEIGHT / original_height
        width_scale = NORMALIZED_WIDTH / original_width

        scale = min(height_scale, width_scale)
        new_height = round(original_height * scale)
        new_width = round(original_width * scale)

        resized_character = cv2.resize(
            character_binary,
            (new_width, new_height)
        )

        resized_characters.append(resized_character)
        print(
            f"Character {index}: "
            f"original={character_binary.shape}, "
            f"resized={resized_character.shape}"
        )

    print("Stored characters:", len(resized_characters))

    normalized_characters = []
    for resized_character in resized_characters:
        canvas = np.full(
            (NORMALIZED_HEIGHT, NORMALIZED_WIDTH),
            255,
            dtype = np.uint8
        )

        x_start = (NORMALIZED_WIDTH - resized_character.shape[1]) // 2
        x_end = x_start + resized_character.shape[1]

        y_start = (NORMALIZED_HEIGHT - resized_character.shape[0]) // 2
        y_end = y_start + resized_character.shape[0]

        canvas[y_start:y_end, x_start:x_end] = resized_character
        normalized_characters.append(canvas)

    for index, normalized in enumerate(normalized_characters):
        print(f"Character {index}: {normalized.shape}")

    preview = np.full(
        (NORMALIZED_HEIGHT, NORMALIZED_WIDTH * len(normalized_characters)),
        255,
        dtype = np.uint8
    )

    for i, normalized_character in enumerate(normalized_characters):
        x_start = NORMALIZED_WIDTH * i
        x_end = x_start + NORMALIZED_WIDTH

        preview[:, x_start:x_end] = normalized_character

    templates = {
    "2": [normalized_characters[0]],
    "9": [
        normalized_characters[1],
        normalized_characters[4],
        normalized_characters[5],
        normalized_characters[6],
        normalized_characters[7],
        normalized_characters[8],
    ],
    "C": [normalized_characters[2]],
    "1": [normalized_characters[3]],
}
    
    plate_string = ""
    for index, normalized_character in enumerate(normalized_characters):
        label, score = predict_character(normalized_character, templates)

        print(
            f"Character {index}: "
            f"{label}, score={score:.4f}"
        )

        if label is not None:
            plate_string += label

    print(plate_string)
    # --------------------------------------------------
    # Draw accepted contours
    # --------------------------------------------------

    # Use a copy so the original cropped image
    # is not modified.
    contour_view = plate_crop.copy()

    cv2.drawContours(
        contour_view,
        character_contours,
        contourIdx=-1,       # Draw every contour in the list.
        color=(0, 255, 0),   # Green in BGR format.
        thickness=2
    )

    
    # --------------------------------------------------
    # Draw box view
    # --------------------------------------------------
    box_view = plate_crop.copy()

    for contour in character_contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        cv2.rectangle(
            box_view,
            (x, y),
            (x + box_width, y + box_height),
            (0, 255, 0),
            2
        )

    # --------------------------------------------------
    # Draw unique box view
    # --------------------------------------------------
    unique_box_view = plate_crop.copy()
    
    for x, y, box_width, box_height in unique_boxes:
        cv2.rectangle(
            unique_box_view,
            (x, y),
            (x + box_width, y + box_height),
            (0, 255, 0),
            2
        )



    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    # cv2.imshow("Original Plate Crop", plate_crop)
    # cv2.imshow("Canny Edges", edges)
    # cv2.imshow("Closed Edges", closed_edges)
    # cv2.imshow("Detected Character Contours", contour_view)
    # cv2.imshow("Character Boxes", box_view)
    cv2.imshow("Unique Character Boxes", unique_box_view)
    cv2.imshow("Character Preview", preview)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
