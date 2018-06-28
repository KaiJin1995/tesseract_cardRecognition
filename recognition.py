import pytesseract
from PIL import Image


def recognition_card(img):
    image = Image.fromarray(img)
    tessdata_dir_config = '--psm 7 --oem 1'
    result = pytesseract.image_to_string(image, lang='eng', config=tessdata_dir_config)
    return result






