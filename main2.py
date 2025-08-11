import cv2 
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils
import re

# List of valid Indian State and Union Territory codes
INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DH", "DL", "GA", "GJ", "HR", "HP", 
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", 
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UA", "UK", "UP", "WB", "AN", "DN", "BH"
}

# Load the image (replace 'car_image.jpeg' with your image file)
img = cv2.imread('car_image.jpeg') 
if img is None:
    print("Error: Image not found. Please check the file path.")
    exit()

# Display the original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Image preprocessing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # Noise reduction
plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB))
plt.title('Processed Image')
plt.show()
edged = cv2.Canny(bfilter, 30, 200) # Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title('Edge Detection')
plt.show()

# Find contours and locate the license plate
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

print("Location: ", location)

if location is not None:
    # Masking and cropping the license plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1) 
    new_image = cv2.bitwise_and(img, img, mask=mask) 
    
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.title('Masked Image')
    plt.show()
    
    (x, y) = np.where(mask == 255) 
    (x1, y1) = (np.min(x), np.min(y)) 
    (x2, y2) = (np.max(x), np.max(y)) 
    cropped_image = gray[x1:x2+1, y1:y2+1]

    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')
    plt.show()
    
    # Use EasyOCR to read text
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    
    # --- Post-processing and validation logic ---
    final_text = "No Plate Detected"
    
    if result:
        raw_text = result[0][-2].upper().replace(' ', '')
        print(f"EasyOCR Raw Text: {raw_text}")
        
        # 1. Character correction for common OCR errors (Z->2, G->6, I->1, O->0)
        corrected_text = raw_text.replace('Z', '2').replace('G', '6').replace('I', '1').replace('O', '0')
        
        # 2. Check and correct the first two characters (State code)
        state_code_from_ocr = corrected_text[:2]
        
        if state_code_from_ocr in INDIAN_STATE_CODES:
            final_state_code = state_code_from_ocr
        else:
            print(f"Invalid State Code '{state_code_from_ocr}' detected. Attempting correction...")
            # Simple correction for common misreads
            if state_code_from_ocr == 'MW':
                final_state_code = 'MH'
            elif state_code_from_ocr == 'HK':
                final_state_code = 'HR'
            elif state_code_from_ocr == 'UR':
                final_state_code = 'UP'
            else:
                final_state_code = state_code_from_ocr
        
        # Combine the corrected state code with the rest of the plate
        plate_without_state_code = corrected_text[2:]
        combined_text = final_state_code + plate_without_state_code

        # 3. Format validation and correction using Regex
        # Pattern to capture and format the parts of the plate
        pattern = re.compile(r'([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{1,4})')
        match = pattern.search(combined_text)
        
        if match:
            state_code = match.group(1).strip()
            rto_code = match.group(2).strip().zfill(2) # Pad with leading zeros
            series_code = match.group(3).strip()
            unique_number = match.group(4).strip().zfill(4) # Pad with leading zeros
            
            final_text = f"{state_code} {rto_code} {series_code} {unique_number}"
        else:
            print("Warning: Plate format could not be fully validated. Displaying corrected text.")
            final_text = corrected_text
            
    else:
        print("No text detected in the cropped image.")

    print(f"Final Detected Text: {final_text}")
    
    # Display the final image with the detected text
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=final_text, org=(location[0][0][0], location[1][0][1] + 60), 
                      fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)
    
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.title('Final Image with Text')
    plt.show()

else:
    print("Could not find a license plate in the image.")