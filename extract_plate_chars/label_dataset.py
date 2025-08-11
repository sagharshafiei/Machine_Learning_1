# label_dataset.py
def main():
    import os, cv2, pandas as pd
    from pathlib import Path

    CROPPED_FOLDER = "plate_chars"
    OUTPUT_EXCEL   = "final_dataset.xlsx"
    TRASH_FOLDER   = "plate_noise"   # برای موارد مزاحم (مثلاً I.R. IRAN)
    Path(TRASH_FOLDER).mkdir(exist_ok=True)
    if not os.path.isdir(CROPPED_FOLDER):
        raise FileNotFoundError("پوشه plate_chars نیست. اول extract را اجرا کن.")

    # لیست یکتا از فایل‌ها
    files = sorted({f for f in os.listdir(CROPPED_FOLDER) if f.lower().endswith(".png")})

    # دیتافریم خروجی با ستون وضعیت
    if os.path.exists(OUTPUT_EXCEL):
        df = pd.read_excel(OUTPUT_EXCEL, dtype=str)
    else:
        df = pd.DataFrame(columns=["filename","label","path","status"])  # status: labeled/skip/noise

    done = set(df["filename"].astype(str)) if not df.empty else set()
    todo = [f for f in files if f not in done]
    print(f"✅ قبلاً ثبت شده: {len(done)} | 🖼️ باقی‌مانده: {len(todo)}")

    if not todo:
        print("همه تصاویر قبلاً ثبت شده‌اند.")
        return

    cv2.namedWindow("Label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label", 320, 320)

    HELP = """
[راهنما]
0..9  → ثبت سریع رقم انگلیسی
T     → تایپ آزاد (فارسی/یونیکد)
S     → Skip (رد کردن؛ دیگر نشان داده نمی‌شود)
N     → Noise (مزاحم؛ انتقال فایل به plate_noise و ثبت نمی‌شود)
ESC   → خروج امن (تمام ثبت‌ها روی دیسک ذخیره شده)
"""
    print(HELP)

    def save_row(filename, label, path, status):
        nonlocal df
        new = pd.DataFrame([{"filename": filename, "label": label, "path": path, "status": status}])
        df = pd.concat([df, new], ignore_index=True)
        df.drop_duplicates(subset=["filename"], keep="last", inplace=True)  # جلوگیری از تکرار
        df.to_excel(OUTPUT_EXCEL, index=False)

    for fname in todo:
        path = os.path.join(CROPPED_FOLDER, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ نتوانستم {fname} را بخوانم؛ به‌عنوان noise ثبت شد.")
            save_row(fname, "", path, "noise")
            continue

        # نمایش بزرگ و واضح
        show = cv2.resize(img, (180, 180), interpolation=cv2.INTER_CUBIC)
        vis  = cv2.cvtColor(show, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, fname, (5, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        cv2.imshow("Label", vis)

        while True:
            k = cv2.waitKey(0)

            if k == 27:  # ESC
                cv2.destroyAllWindows()
                df.to_excel(OUTPUT_EXCEL, index=False)
                print("🚪 خروج امن. هرچه ثبت شده بود ذخیره شد.")
                return

            # ارقام انگلیسی
            if ord('0') <= k <= ord('9'):
                ch = chr(k)
                save_row(fname, ch, path, "labeled")
                print(f"✅ {fname} → '{ch}'")
                break

            # تایپ آزاد (فارسی/انگلیسی)
            if k in (ord('t'), ord('T')):
                cv2.destroyWindow("Label")   # برای گرفتن فوکوس ترمینال
                label = input(f"📝 برچسب برای {fname} (فارسی هم قابل قبول): ").strip()
                cv2.namedWindow("Label", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Label", 320, 320)
                if label:
                    save_row(fname, label, path, "labeled")
                    print(f"✅ {fname} → '{label}'")
                    break
                else:
                    print("⚠️ خالی بود؛ یکی از کلیدها را بزن (0..9/T/S/N).")
                    cv2.imshow("Label", vis)
                    continue

            # Skip: ثبت می‌شود تا دوباره نشان داده نشود
            if k in (ord('s'), ord('S')):
                save_row(fname, "", path, "skip")
                print(f"⏭️ Skip شد: {fname}")
                break

            # Noise: انتقال به پوشه‌ی مزاحم و ثبت
            if k in (ord('n'), ord('N')):
                new_path = os.path.join(TRASH_FOLDER, fname)
                try:
                    os.replace(path, new_path)  # جابجایی
                except Exception:
                    pass
                save_row(fname, "", new_path, "noise")
                print(f"🗑️ Noise شد و منتقل گردید: {fname}")
                break

            print("⌨️ کلید نامعتبر. از 0..9 یا T/S/N/ESC استفاده کن.")

    cv2.destroyAllWindows()
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"🎉 تمام شد. خروجی: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
