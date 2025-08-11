def main():
    import os
    import pandas as pd
    from PIL import Image, ImageDraw, ImageFont

    FONTS_DIR = "fonts"                 # فونت‌ها را اینجا بگذار
    OUTPUT_DIR = "generated_chars"      # خروجی تصاویر مصنوعی
    OUTPUT_XLSX = "dataset.xlsx"        # خروجی اکسل که مرحله بعدی چک می‌کند

    # کاراکترهایی که می‌خواهی تولید شوند
    # می‌توانی هرکدام را کم/زیاد کنی (اعداد فارسی هم نمونه‌اش آورده شده)
    LATIN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    DIGITS = "0123456789"
    # PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"  # در صورت نیاز فعال کن
    CHARSET = DIGITS + LATIN_UPPER      # + PERSIAN_DIGITS

    IMG_SIZE = (64, 64)
    FONT_SIZE = 48
    POSITIONS = [(8, 8), (12, 8), (8, 12)]  # کمی جابه‌جایی برای تنوع

    if not os.path.isdir(FONTS_DIR):
        raise FileNotFoundError(f"پوشه‌ی فونت '{FONTS_DIR}' پیدا نشد.")

    font_files = [os.path.join(FONTS_DIR, f)
                  for f in os.listdir(FONTS_DIR)
                  if f.lower().endswith((".ttf", ".otf"))]

    if not font_files:
        raise FileNotFoundError("هیچ فونتی در پوشه‌ی fonts پیدا نشد (ttf/otf).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    count = 0

    for font_path in font_files:
        try:
            font = ImageFont.truetype(font_path, FONT_SIZE)
        except Exception:
            # اگر فونت خراب بود یا لود نشد، رد می‌کنیم
            continue

        font_name = os.path.splitext(os.path.basename(font_path))[0]

        for ch in CHARSET:
            # بعضی فونت‌ها ممکنه کاراکتر را پشتیبانی نکنند؛
            # اگر خروجی خالی شد، باز هم تصویرش تولید می‌شود ولی مشکلی نیست.
            for i, (px, py) in enumerate(POSITIONS, start=1):
                img = Image.new("L", IMG_SIZE, 255)  # سفید
                draw = ImageDraw.Draw(img)
                draw.text((px, py), ch, font=font, fill=0)  # مشکی

                # نام فایل امن (بدون حروف خاص)
                code = "-".join(f"U{ord(c):04X}" for c in ch)
                filename = f"{code}_{font_name}_{i}.png"
                out_path = os.path.join(OUTPUT_DIR, filename)
                img.save(out_path)

                rows.append({
                    "filename": filename,
                    "label": ch,
                    "font": font_name,
                    "variant": i,
                    "path": out_path
                })
                count += 1

    pd.DataFrame(rows).to_excel(OUTPUT_XLSX, index=False)
    print(f"✅ {count} تصویر تولید شد و در '{OUTPUT_DIR}' ذخیره گردید.")
    print(f"✅ فایل اکسل دیتاست در '{OUTPUT_XLSX}' ساخته شد.")

if __name__ == "__main__":
    main()
