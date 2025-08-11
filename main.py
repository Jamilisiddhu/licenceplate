import cv2
import imutils
import numpy as np
import easyocr
import re

# This function will correct OCR misreadings and format the plate number
# based on its strict format (2-char, 2-num, 2-char, 4-num).
def correct_ocr_text(text):
    text = text.upper()
    
    # 1. First, remove all non-alphanumeric characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text)

    # 2. Apply a regex pattern to extract the parts
    # This pattern looks for 2 letters, 2 numbers, 2 letters, and 4 numbers
    pattern = re.compile(r'([A-Z]{2})([0-9]{2})([A-Z]{2})([0-9]{4})')
    match = pattern.match(cleaned_text)

    if match:
        state_code = match.group(1)
        district_code = match.group(2)
        series_code = match.group(3)
        unique_number = match.group(4)
        
        # We can apply specific character corrections here if needed, but
        # the regex already enforces the character type.
        
        # Example corrections (if a number is misread as a letter):
        # district_code = district_code.replace('I', '1').replace('O', '0')
        # ...and so on.

        return f"{state_code}{district_code}{series_code}{unique_number}"
    
    # Fallback for when the regex doesn't match perfectly
    # This part will use a more flexible, position-based logic
    
    corrected_parts = []
    
    # Part 1: First 2 chars (Letters)
    if len(cleaned_text) >= 2:
        part1 = cleaned_text[:2]
        part1 = re.sub(r'[^A-Z]', '', part1).zfill(2) # Enforce letters
        corrected_parts.append(part1)
        cleaned_text = cleaned_text[2:]
    
    # Part 2: Next 2 chars (Numbers)
    if len(cleaned_text) >= 2:
        part2 = cleaned_text[:2]
        part2 = re.sub(r'[^0-9]', '', part2).zfill(2) # Enforce numbers
        corrected_parts.append(part2)
        cleaned_text = cleaned_text[2:]
        
    # Part 3: Next 2 chars (Letters)
    if len(cleaned_text) >= 2:
        part3 = cleaned_text[:2]
        part3 = re.sub(r'[^A-Z]', '', part3).zfill(2) # Enforce letters
        corrected_parts.append(part3)
        cleaned_text = cleaned_text[2:]

    # Part 4: Last 4 chars (Numbers)
    if len(cleaned_text) >= 4:
        part4 = cleaned_text[:4]
        part4 = re.sub(r'[^0-9]', '', part4).zfill(4) # Enforce numbers
        corrected_parts.append(part4)
    
    return "".join(corrected_parts)


# Load the image from your project folder
image = cv2.imread('car_image.jpeg')

# Check if the image was loaded correctly
if image is None:
    print("Error: Could not read the image file.")
    print("Please check the file path and name.")
    exit()

# Resize the image for consistent processing
image = imutils.resize(image, width=500)

# Convert the image to grayscale to simplify color information
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a bilateral filter to reduce noise while keeping edges sharp
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Find edges in the grayscale image using the Canny algorithm
edged = cv2.Canny(gray, 170, 200)

# Find all contours (outlines of shapes) in the edged image
contours, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by their area, from largest to smallest, and get the top 30
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Initialize a variable to hold the location of the license plate
license_plate_contour = None

# Loop through the sorted contours
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        
        if 3 < aspect_ratio < 5:
            license_plate_contour = approx
            break

if license_plate_contour is not None:
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [license_plate_contour], 0, 255, -1)
else:
    print("Could not detect a license plate in the image.")
    exit()

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped_plate = gray[topx:bottomx+1, topy:bottomy+1]

# --- EASYOCR PART ---
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_plate)
text = ""

if result:
    detected_texts = []
    for res in result:
        text_to_check = res[1].upper().strip()
        if len(text_to_check) > 2 and 'IND' not in text_to_check:
            detected_texts.append(res[1])
            
    combined_text = ' '.join(detected_texts)
    
    # Process the text to strictly adhere to the specified format
    final_text = correct_ocr_text(combined_text)
    
    print(f"EasyOCR raw result: {[res[1] for res in result]}")
    print(f"Combined text for processing: {combined_text}")
    print(f"Detected License Plate Number: {final_text}")

else:
    print("No text was detected by EasyOCR.")
    final_text = ""
    
# --- DISPLAYING THE FINAL RESULT ---
final_image = cv2.drawContours(image.copy(), [license_plate_contour], -1, (0, 255, 0), 3)

if license_plate_contour is not None and len(license_plate_contour) > 0:
    cv2.putText(final_image, final_text, (license_plate_contour[0][0][0], license_plate_contour[0][0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("License Plate Detection", final_image)
print(f"Detected License Plate Number: {final_text}")

cv2.waitKey(0)
cv2.destroyAllWindows()