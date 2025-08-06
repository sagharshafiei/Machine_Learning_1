# extract_plate_chars.py

def main():
    import cv2
    import os
    import uuid

    IMAGE_PATH = "images/plate.png"
    OUTPUT_FOLDER = "plate_chars"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"تصویر '{IMAGE_PATH}' پیدا نشد.")

    img = cv2.pyrUp(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"🔍 تعداد کل کانتور پیدا شده: {len(contours)}")
    char_count = 0
    skipped = 0

    img_copy = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000 < area < 8000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # نمایش تصویر با کانتورهای مشخص‌شده
    cv2.imshow("شناسایی کانتورها", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2000 < area < 8000:
            x, y, w, h = cv2.boundingRect(cnt)
            char_img = gray[y:y+h, x:x+w]
            char_img = cv2.resize(char_img, (28, 28))

            file_name = f"char_{uuid.uuid4().hex[:8]}.png"
            file_path = os.path.join(OUTPUT_FOLDER, file_name)
            cv2.imwrite(file_path, char_img)
            char_count += 1
        else:
            skipped += 1

    print(f"✅ {char_count} کاراکتر ذخیره شد.")
    print(f"🚫 {skipped} کانتور رد شد چون خارج از محدوده مساحت بود.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
