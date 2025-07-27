import fitz  # PyMuPDF
import os
from tqdm import tqdm

PDF_DIR = "training_pdfs"
OUTPUT_DIR = "text_dumps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text += f"\n--- Page {page_num} ---\n{text}"
    return full_text

def main():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    for pdf_file in tqdm(pdf_files, desc="Extracting text"):
        input_path = os.path.join(PDF_DIR, pdf_file)
        output_path = os.path.join(OUTPUT_DIR, pdf_file.replace(".pdf", ".txt"))

        text = extract_text_from_pdf(input_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"\n✅ Extracted text for {len(pdf_files)} PDFs. Saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
