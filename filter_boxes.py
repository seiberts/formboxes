import cv2
import numpy as np

def non_maximum_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def detect_checkboxes(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # Invert for white boxes on dark background

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect boxes and apply non-maximum suppression
    boxes = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Filter for square-like boxes
            if 0.9 <= aspect_ratio <= 1.1:
                boxes.append((x, y, x + w, y + h))

    # Apply Non-Maximum Suppression
    boxes = non_maximum_suppression(boxes, 0.3)

    # Check for markings and draw results
    for (x1, y1, x2, y2) in boxes:
        roi = gray[y1:y2, x1:x2]
        non_white_pixels = cv2.countNonZero(roi)
        
        if non_white_pixels > 50:  # Adjust threshold if needed
            cv2.putText(image, "Marked", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Unmarked", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw the bounding boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the final result
    cv2.imshow('Detected Checkboxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your image
image_path = 'path_to_your_image.png'
detect_checkboxes(image_path)
