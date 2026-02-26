import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract
import os

# Set the path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Load the image
# image_path = r'C:\Users\Venkatesh UR\Downloads\HEMI MINI\license plate000\input01.jpg'
# img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# if img is None:
#     print(f"Error: Unable to read the image at {image_path}. Check the file path and integrity.")
#     exit()



image_path = r"C:\Users\dhanu\Desktop\lpd\input01.png"
# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: File not found at {image_path}")
    exit()

# Try reading the image
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img is None:
    print(f"Error: Unable to read the image at {image_path}. Check the file path and integrity.")
    exit()

# Display the original image
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()

# Resize the image
img = imutils.resize(img, width=500)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter to reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Perform edge detection
edged = cv2.Canny(gray, 30, 200)

# Display edge detection result
plt.figure(figsize=(10, 5))
plt.imshow(edged, cmap='gray')
plt.title("Edge Detection")
plt.show()

# Find contours
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours
img1 = img.copy()
cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("All Contours")
plt.show()

# Sort contours by area
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None

# Loop over contours to find the license plate
for c in cnts:
    # Approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
    # Look for a contour with four points
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    print("Error: No contour detected.")
    exit()

# Draw the detected license plate contour
cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

# Create a mask and extract the license plate
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [screenCnt], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Crop the license plate region
(x, y) = np.where(mask == 255)
if x.size == 0 or y.size == 0:
    print("Error: Unable to crop license plate. Check contour detection.")
    exit()

(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

# Display cropped license plate
plt.figure(figsize=(10, 5))
plt.imshow(Cropped, cmap='gray')
plt.title("Cropped License Plate")
plt.show()

# Perform OCR to extract text
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Programming Fever's License Plate Recognition")
print("Detected license plate number is:", text)

# Resize and display the original and cropped images
img = cv2.resize(img, (500, 300))
Cropped = cv2.resize(Cropped, (400, 200))