# label_dataset.py

def main():
    import cv2
    import os
    import pandas as pd

    DATASET_EXCEL = "dataset.xlsx"
    CROPPED_FOLDER = "plate_chars"
    OUTPUT_EXCEL = "final_dataset.xlsx"

    if not os.path.exists(DATASET_EXCEL):
        raise FileNotFoundError("فایل dataset.xlsx پیدا نشد. اول generate_fonts_dataset.py را اجرا کن.")

    if not os.path.exists(CROPPED_FOLDER):
        raise FileNotFoundError("پوشه‌ی plate_chars پیدا نشد. اول extract_plate_chars.py را اجرا کن.")

    cropped_files = [f for f in os.listdir(CROPPED_FOLDER) if f.endswith(".png")]

    final_data = []

    for img_file in cropped_files:
        full_path = os.path.join(CROPPED_FOLDER, img_file)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        cv2.imshow("Character", img)
        cv2.waitKey(1)
        label = input(f"📝 لطفاً برچسب این کاراکتر ({img_file}) را وارد کن: ").strip()
        cv2.destroyAllWindows()

        if label:
            final_data.append({
                "filename": img_file,
                "label": label,
                "path": full_path
            })

    pd.DataFrame(final_data).to_excel(OUTPUT_EXCEL, index=False)
    print(f"✅ دیتاست نهایی با {len(final_data)} نمونه ذخیره شد در {OUTPUT_EXCEL}.")

if __name__ == "__main__":
    main()
