import fitz
import os

pdf_path = "input.pdf"
output_dir = "extracted_images"

os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)

saved = 0

for page_num in range(len(doc)):
    page = doc[page_num]
    images = page.get_images(full=True)

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image["ext"]

        out_path = os.path.join(
            output_dir,
            f"page_{page_num + 1}_img_{img_index + 1}.{ext}"
        )

        with open(out_path, "wb") as f:
            f.write(image_bytes)

        print(f"Saved: {out_path}")
        saved += 1

print(f"Done. Extracted {saved} images.")