import cv2
import numpy as np

def create_mask(image):
    """
    Create a binary mask by drawing a polygon around the object.
    
    Args:
        image (numpy array): The input image.
    
    Returns:
        mask (numpy array): A binary mask where the ROI is set to 1 and the background is set to 0.
    """
    img_copy = image.copy()
    coords = []
    drawing = False

    def draw_shape(event, x, y, flags, param):
        nonlocal coords, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            coords.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            coords.append((x, y))
            cv2.polylines(img_copy, [np.int32(coords)], True, (0, 255, 0), 2)

    cv2.namedWindow("Image Editor")
    cv2.setMouseCallback("Image Editor", draw_shape)

    print("Draw a polygon around the object to create a mask.")
    print("Left-click to start drawing, move the mouse to draw, and left-click again to finish.")
    print("Press 'c' to confirm the selection or 'r' to reset.")

    while True:
        cv2.imshow("Image Editor", img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break
        elif key == ord("r"):
            img_copy = image.copy()
            coords = []
            drawing = False

    coords = np.int32(coords)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [coords], -1, 1, -1)

    return mask

img = cv2.imread("nameLogo.png")
mask = create_mask(img)

# Initialize the mask with GC_BGD (0) and set the ROI to GC_FGD (1)
mask[mask == 255] = 1
mask[mask == 0] = 0

# Apply GrabCut to remove the background
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

# Refine the mask
mask, bgd_model, fgd_model = cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_EVAL)

# Apply the mask to the original image
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
