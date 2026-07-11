import cv2
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_PATH = "plate.jpg"

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


# --------------------------------------------------
# Load the original image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Could not read image: {IMAGE_PATH}")

else:
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
        CROP_Y_START:CROP_Y_END,
        CROP_X_START:CROP_X_END
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
    # Display results
    # --------------------------------------------------

    # cv2.imshow("Original Plate Crop", plate_crop)
    # cv2.imshow("Canny Edges", edges)
    # cv2.imshow("Closed Edges", closed_edges)
    cv2.imshow("Detected Character Contours", contour_view)
    cv2.imshow("Character Boxes", box_view)

    cv2.waitKey(0)
    cv2.destroyAllWindows()