import os
import fitz  # PyMuPDF
import csv
import argparse
from tqdm import tqdm

def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    data = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue  # skip non-text blocks
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text or len(text) > 50:
                        continue

                    features = {
                        "text": text,
                        "font_size": span.get("size", 0),
                        "bold": int("bold" in span.get("font", "").lower()),
                        "italic": int("italic" in span.get("font", "").lower() or "oblique" in span.get("font", "").lower()),
                        "x0": span["bbox"][0],
                        "y0": span["bbox"][1],
                        "x1": span["bbox"][2],
                        "y1": span["bbox"][3],
                        "page": page_num
                    }
                    data.append(features)
    return data

def save_to_csv(data, output_csv_path):
    if not data:
        return
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)

def convert_pdfs_to_csvs(pdf_dir, csv_dir):
    os.makedirs(csv_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        csv_filename = os.path.splitext(pdf_file)[0] + ".csv"
        csv_path = os.path.join(csv_dir, csv_filename)

        data = extract_features_from_pdf(pdf_path)
        save_to_csv(data, csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from PDFs and save as CSVs.")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Path to directory with PDFs")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to save CSVs")
    args = parser.parse_args()

    convert_pdfs_to_csvs(args.pdf_dir, args.csv_dir)
