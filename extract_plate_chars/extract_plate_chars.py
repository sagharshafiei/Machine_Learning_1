# extract_plate_chars.py
def main():
    import os
    import uuid
    import cv2
    import numpy as np

    # ===== تنظیمات اصلی =====
    IMAGE_PATH = "images/plate.png"     # مسیر تصویر پلاک
    OUTPUT_FOLDER = "plate_chars"       # خروجی برش کاراکترها
    DEBUG_FOLDER = "debug"              # ذخیره پیش‌نمایش
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

    # نمایش پنجره‌های دیباگ (برای خاموش کردن بگذار False)
    SHOW_DEBUG = True

    # بزرگ‌نمایی برای تصاویر کوچک (1.0 یعنی بدون تغییر)
    SCALE = 2.0

    # روش باینری‌کردن: ADAPTIVE یا OTSU
    USE_ADAPTIVE = True

    # روش جداکردن اجزا: Connected Components (پیش‌فرض) یا Contours
    USE_CONNECTED_COMPONENTS = True

    # ===== فیلترهای نسبی برای انتخاب باکس‌های کاراکتر =====
    # (به‌صورت نسبت به اندازه تصویر؛ از اعداد ثابت بهتر جواب می‌دهد)
    MIN_AREA_RATIO = 0.0002   # حداقل نسبت مساحت باکس به مساحت کل تصویر (0.02%)
    MAX_AREA_RATIO = 0.20     # حداکثر نسبت مساحت باکس (20%)

    MIN_H_RATIO = 0.10        # حداقل نسبت ارتفاع باکس به ارتفاع تصویر
    MAX_H_RATIO = 0.98        # حداکثر نسبت ارتفاع باکس

    MIN_AR = 0.08             # حداقل نسبت عرض/ارتفاع (برای کاراکترهای باریک مثل 1/I)
    MAX_AR = 2.5              # حداکثر نسبت عرض/ارتفاع (برای W/M)

    # پارامترهای Adaptive Threshold (در صورت USE_ADAPTIVE=True)
    ADAPT_BLOCK_SIZE = 31     # باید فرد باشد: 21، 31، 35 ...
    ADAPT_C = 5               # بایاس؛ 3 تا 9 تست خوبی دارد

    # ===== خواندن و آماده‌سازی تصویر =====
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"تصویر '{IMAGE_PATH}' پیدا نشد.")

    if SCALE and SCALE != 1.0:
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ===== باینری کردن =====
    if USE_ADAPTIVE:
        # کاراکترها سفید، پس‌زمینه سیاه (INV)
        thresh = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPT_BLOCK_SIZE, ADAPT_C
        )
    else:
        _, thresh = cv2.threshold(
            gray_blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # ===== مورفولوژی (کم، برای اتصال شکاف‌های ریز) =====
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # اگر حروف تکه‌تکه‌اند، CLOSE را بزرگ‌تر کن (مثلاً (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    H, W = gray.shape[:2]
    img_area = float(H * W)

    # ===== انتخاب کاندیدها =====
    candidates = []

    if USE_CONNECTED_COMPONENTS:
        # روش پایدارتر نسبت به کانتور
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        # stats[i] = [x, y, w, h, area]
        for i in range(1, numLabels):  # 0 پس‌زمینه است
            x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
            area_ratio = area / img_area
            h_ratio = h / float(H)
            ar = (w / float(h)) if h > 0 else 0.0

            if (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO and
                MIN_H_RATIO   <= h_ratio    <= MAX_H_RATIO   and
                MIN_AR        <= ar         <= MAX_AR):
                candidates.append((x, y, w, h))
    else:
        # روش کانتور (در صورت نیاز)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            area_ratio = area / img_area
            h_ratio = h / float(H)
            ar = (w / float(h)) if h > 0 else 0.0
            if (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO and
                MIN_H_RATIO   <= h_ratio    <= MAX_H_RATIO   and
                MIN_AR        <= ar         <= MAX_AR):
                candidates.append((x, y, w, h))

    # اگر هیچ کاندیدی پیدا نشد، یک‌بار دیگر با معیار شل‌تر و OTSU تلاش می‌کنیم
    if len(candidates) == 0 and USE_ADAPTIVE:
        # سوئیچ به Otsu + فیلتر شل‌تر
        _, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2, iterations=1)

        # فیلترهای شل‌تر
        _MIN_AREA_RATIO = max(MIN_AREA_RATIO * 0.5, 0.00005)
        _MAX_AREA_RATIO = min(MAX_AREA_RATIO * 1.5, 0.6)
        _MIN_H_RATIO = max(MIN_H_RATIO * 0.5, 0.05)
        _MAX_H_RATIO = 0.99
        _MIN_AR = max(MIN_AR * 0.5, 0.04)
        _MAX_AR = MAX_AR * 1.5

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            area_ratio = area / img_area
            h_ratio = h / float(H)
            ar = (w / float(h)) if h > 0 else 0.0
            if (_MIN_AREA_RATIO <= area_ratio <= _MAX_AREA_RATIO and
                _MIN_H_RATIO   <= h_ratio    <= _MAX_H_RATIO   and
                _MIN_AR        <= ar         <= _MAX_AR):
                candidates.append((x, y, w, h))

    # مرتب‌سازی چپ به راست (برای نظم بهتر)
    candidates.sort(key=lambda b: b[0])

    # ===== پیش‌نمایش =====
    preview = img.copy()
    for (x, y, w, h) in candidates:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if SHOW_DEBUG:
        cv2.imshow("thresh", thresh)
        cv2.imshow("candidates", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(DEBUG_FOLDER, "extract_preview.png"), preview)

    # ===== برش، نرمال‌سازی مربعی، تغییر اندازه به 28x28 =====
    def crop_to_square(binary_img, x, y, w, h, out_size=28):
        roi = binary_img[y:y+h, x:x+w]  # باینری: کاراکتر سفید، زمینه سیاه
        h2, w2 = roi.shape[:2]
        side = max(h2, w2)
        pad_top = (side - h2) // 2
        pad_bottom = side - h2 - pad_top
        pad_left = (side - w2) // 2
        pad_right = side - w2 - pad_left
        roi_sq = cv2.copyMakeBorder(
            roi, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0  # سیاه
        )
        roi_sq = cv2.resize(roi_sq, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return roi_sq

    saved = 0
    for (x, y, w, h) in candidates:
        char_img = crop_to_square(thresh, x, y, w, h, out_size=28)
        file_name = f"char_{uuid.uuid4().hex[:8]}.png"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, file_name), char_img)
        saved += 1

    # آمار
    # اگر از connected components استفاده شد، تعداد کل اجزا را همینجا تخمین نمی‌زنیم.
    # یک عدد تقریبی با کانتورها می‌دهیم تا حس بهتری بدهد:
    try:
        contours_all, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_found = len(contours_all)
    except Exception:
        total_found = len(candidates)

    skipped = max(total_found - len(candidates), 0)

    print(f"🔍 کانتورهای/اجزای یافت‌شده (تقریبی): {total_found}")
    print(f"✅ کاراکترهای ذخیره‌شده: {saved}  →  پوشه: '{OUTPUT_FOLDER}'")
    print(f"🚫 موارد ردشده: {skipped}")
    if saved == 0:
        print("⚠️ هیچ کاراکتری ذخیره نشد. SCALE را بیشتر کن یا آستانه‌ها را شل‌تر کن (MIN_AREA_RATIO↓، MAX_AREA_RATIO↑).")

if __name__ == "__main__":
    main()
