import cv2
import pytesseract
from PIL import Image, ImageOps
import os
import shutil
import pandas as pd
import numpy as np
import re


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    
    image = Image.open(image_path)

    
    grayscale_image = image.convert('L')

    
    inverted_image = ImageOps.invert(grayscale_image)

    
    _, thresholded_image = cv2.threshold(np.array(inverted_image), 128, 255, cv2.THRESH_BINARY)

    
    thresholded_image_pil = Image.fromarray(thresholded_image)

    return thresholded_image_pil

def capture_visiting_card():
    
    cap = cv2.VideoCapture(0)

    
    cv2.waitKey(1000)

    
    while True:
        ret, frame = cap.read()
        cv2.imshow("Visiting Card Preview", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    ret, frame = cap.read()

    
    image_path = "visiting_card.jpg"
    cv2.imwrite(image_path, frame)

    
    cap.release()
    cv2.destroyAllWindows()

    return image_path

def extract_text_from_image(image_path):
    
    preprocessed_image = preprocess_image(image_path)

    
    extracted_text = pytesseract.image_to_string(preprocessed_image)

    return extracted_text

def save_image_and_text(image_path, extracted_text):
    
    sorted_folder = "SortedImages"
    if not os.path.exists(sorted_folder):
        os.makedirs(sorted_folder)

    
    words = [word for word in extracted_text.split() if word.isalpha()]
    first_two_words = ' '.join(words[:2])

    
    filename = f"{first_two_words.lower().replace(' ', '_')}.jpg"

    
    sorted_path = os.path.join(sorted_folder, filename)

    try:
        
        shutil.copy(image_path, sorted_path)

        
        excel_path = os.path.join(sorted_folder, "extracted_data.xlsx")
        
        
        if os.path.exists(excel_path):
            
            existing_data = pd.read_excel(excel_path)
            
            new_data = pd.DataFrame({"File Name": [filename], "Extracted Text": [extracted_text]})
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            
            updated_data = pd.DataFrame({"File Name": [filename], "Extracted Text": [extracted_text]})

        
        updated_data.to_excel(excel_path, index=False)
        
        print("Image and extracted text moved to SortedImages folder:", sorted_path)
        print("Extracted text saved to Excel:", excel_path)

    except Exception as e:
        print("Error moving the image or saving extracted text:", e)

if __name__ == "__main__":
    
    captured_card_path = capture_visiting_card()
    print(f"Image captured and saved at: {captured_card_path}")

    
    extracted_text = extract_text_from_image(captured_card_path)
    
    if not extracted_text.strip():
        print("Text extraction failed. Please check the OCR configuration.")
    else:
        print("Extracted Text:", extracted_text)

        
        save_image_and_text(captured_card_path, extracted_text)
