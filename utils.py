import fitz  # PyMuPDF

def extract_text_elements(pdf_path):
    doc = fitz.open(pdf_path)
    elements = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if len(text) == 0 or len(text) > 30:
                        continue
                    elements.append({
                        "text": text,
                        "font_size": span.get("size", 0),
                        "bold": "bold" in span.get("font", "").lower(),
                        "italic": "italic" in span.get("font", "").lower(),
                    })
    return elements
