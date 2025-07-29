import fitz  # pip install pymupdf

def extract_with_layout(path):
    doc = fitz.open(path)
    pages = []
    for page in doc:
        blocks = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0: 
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    blocks.append({
                        "text": span["text"],
                        "bbox": span["bbox"]  # [x0, y0, x1, y1]
                    })
        pages.append(blocks)
    return pages