import os
import sys
import json
import fitz  # PyMuPDF
import pandas as pd
from joblib import load
import numpy as np

MODEL_PATH = "model/heading_model.joblib"
LABEL_MAP_PATH = "model/label_map.json"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    data = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text or len(text) > 80:
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
    
    doc.close()
    return data

def enhance_features(df):
    """Add enhanced features to match training data"""
    
    # Text-based features
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["is_uppercase"] = df["text"].str.isupper().astype(int)
    df["is_titlecase"] = df["text"].str.istitle().astype(int)
    df["ends_with_punctuation"] = df["text"].str.rstrip().str.endswith(('.', '!', '?', ':')).astype(int)
    df["starts_with_number"] = df["text"].str.lstrip().str.split().str[0].str.replace('.', '').str.isdigit().fillna(False).astype(int)
    df["has_colon"] = df["text"].str.contains(':').astype(int)
    
    # Position-based features (relative to page)
    df["relative_x"] = df.groupby("page")["x"].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
    df["relative_y"] = df.groupby("page")["y"].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
    
    # Font size relative features
    df["font_size_rank"] = df.groupby("page")["font_size"].rank(method="dense", ascending=False)
    df["is_largest_font"] = (df["font_size_rank"] == 1).astype(int)
    df["font_size_percentile"] = df.groupby("page")["font_size"].transform(lambda x: x.rank(pct=True))
    
    # Text pattern features
    df["has_digits"] = df["text"].str.contains(r'\d').astype(int)
    df["has_special_chars"] = df["text"].str.contains(r'[^\w\s]').astype(int)
    df["starts_with_capital"] = df["text"].str.match(r'^[A-Z]').fillna(False).astype(int)
    
    return df

def load_model_and_mapping():
    try:
        model = load(MODEL_PATH)
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        reverse_map = {v: k for k, v in label_map.items()}
        
        # Try to load feature list if available
        features_path = "model/features.json"
        expected_features = None
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                expected_features = json.load(f)
        
        return model, reverse_map, expected_features
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def predict_headings(df, model, reverse_map, expected_features=None):
    # Rename columns to match training data
    df = df.rename(columns={"x0": "x", "y0": "y"})
    df = df.drop(columns=["x1", "y1"], errors="ignore")

    # Determine which features to use based on model
    if expected_features is not None:
        # Use features from saved feature list
        feature_cols = expected_features
        print(f"🔧 Using saved feature list: {len(feature_cols)} features")
    else:
        # Try to detect model type by attempting prediction with basic features first
        basic_features = ["font_size", "bold", "italic", "x", "y", "page"]
        
        # Check if model expects basic features only
        try:
            test_X = df[basic_features].head(1).astype({
                "font_size": float, "bold": int, "italic": int,
                "x": float, "y": float, "page": int
            })
            _ = model.predict(test_X)  # Test prediction
            feature_cols = basic_features
            print(f"🔧 Using basic features (legacy model): {feature_cols}")
        except:
            # Model expects enhanced features, so add them
            df = enhance_features(df)
            feature_cols = [
                "font_size", "bold", "italic", "x", "y", "page",
                "text_length", "word_count", "is_uppercase", "is_titlecase",
                "ends_with_punctuation", "starts_with_number", "has_colon",
                "relative_x", "relative_y", "font_size_rank", "is_largest_font",
                "font_size_percentile", "has_digits", "has_special_chars", "starts_with_capital"
            ]
            print(f"🔧 Using enhanced features: {len(feature_cols)} features")
    
    # If model expects enhanced features but we don't have them, add them
    if expected_features and any(col not in df.columns for col in expected_features):
        print("🔧 Adding missing enhanced features...")
        df = enhance_features(df)
    
    # Ensure proper data types
    type_mapping = {
        "font_size": float, "bold": int, "italic": int, "x": float, "y": float, "page": int,
        "text_length": int, "is_uppercase": int, "word_count": int, "ends_with_punctuation": int,
        "starts_with_number": int, "has_colon": int, "relative_x": float, "relative_y": float,
        "font_size_rank": float, "is_largest_font": int, "font_size_percentile": float,
        "has_digits": int, "is_titlecase": int, "has_special_chars": int, "starts_with_capital": int
    }
    
    for col in feature_cols:
        if col in df.columns and col in type_mapping:
            df[col] = df[col].astype(type_mapping[col])

    X = df[feature_cols]
    
    # Debug: Print feature statistics
    print(f"\n📊 Using {len(feature_cols)} features: {feature_cols}")
    print("Feature statistics:")
    print(X.describe())
    
    preds = model.predict(X)
    probas = model.predict_proba(X)

    print(f"\n🧠 Raw class predictions: {preds.tolist()}")
    print("📊 Prediction class distribution:")
    print(pd.Series(preds).value_counts().sort_index())
    
    # Check class mapping
    print(f"\n🗺️ Label mapping: {reverse_map}")

    df["predicted_label"] = preds
    df["predicted_label"] = df["predicted_label"].map(reverse_map)

    # Log top 3 predicted probabilities for first few rows
    print("\n🔍 Sample prediction confidences:")
    for i in range(min(10, len(df))):
        probs = probas[i]
        # Get class names correctly
        class_names = [reverse_map.get(j, f"class_{j}") for j in range(len(probs))]
        top = sorted(zip(probs, class_names), reverse=True)[:3]
        print(f"{df.iloc[i]['text'][:30]:30} → {[f'{label}:{p:.3f}' for p, label in top]}")

    return df

def simple_heuristic_classification(df):
    """
    Fallback heuristic classification when model fails
    """
    print("\n🛠️ Applying heuristic classification...")
    
    df["predicted_label"] = "normal"
    
    # Sort by page then by y position (top to bottom)
    df_sorted = df.sort_values(['page', 'y0']).reset_index(drop=True)
    
    # Find potential title (first large text on first page)
    first_page = df_sorted[df_sorted['page'] == 0]
    if not first_page.empty:
        # Look for title characteristics
        title_candidates = first_page[
            (first_page['font_size'] >= first_page['font_size'].quantile(0.8)) |
            (first_page['bold'] == 1) |
            (first_page['text_length'] <= 50)
        ].head(3)
        
        if not title_candidates.empty:
            # Pick the first one as title
            title_idx = title_candidates.index[0]
            df.loc[title_idx, 'predicted_label'] = 'TITLE'
            print(f"📝 Identified title: {df.loc[title_idx, 'text']}")
    
    # Find potential headings based on formatting
    for idx, row in df.iterrows():
        if row['predicted_label'] != 'TITLE':
            # Heading heuristics
            is_large_font = row['font_size'] >= df['font_size'].quantile(0.7)
            is_bold = row['bold'] == 1
            is_short = row['text_length'] <= 60
            is_sentence_case = not row['text'].islower()
            
            heading_score = sum([is_large_font, is_bold, is_short, is_sentence_case])
            
            if heading_score >= 2:
                if row['font_size'] >= df['font_size'].quantile(0.9):
                    df.loc[idx, 'predicted_label'] = 'H1'
                elif row['font_size'] >= df['font_size'].quantile(0.8):
                    df.loc[idx, 'predicted_label'] = 'H2'
                else:
                    df.loc[idx, 'predicted_label'] = 'H3'
    
    return df

def save_json(df, pdf_path):
    # Find title
    title_rows = df[df["predicted_label"] == "TITLE"]
    title = title_rows["text"].iloc[0] if not title_rows.empty else ""
    
    # If no title found, try to use first heading or first text
    if not title:
        h1_rows = df[df["predicted_label"] == "H1"]
        if not h1_rows.empty:
            title = h1_rows["text"].iloc[0]
        else:
            # Use first text as fallback
            if not df.empty:
                title = df["text"].iloc[0]
    
    # Build outline
    outline = []
    heading_labels = {"H1", "H2", "H3", "H4", "H5"}
    
    for _, row in df.iterrows():
        if row["predicted_label"] in heading_labels:
            outline.append({
                "level": row["predicted_label"],
                "text": row["text"].strip(),
                "page": int(row["page"])
            })

    result = {"title": title.strip(), "outline": outline}
    
    # Pretty print result
    print(f"\n📄 Document Analysis Results:")
    print(f"Title: {result['title']}")
    print(f"Headings found: {len(outline)}")
    for item in outline:
        print(f"  [{item['level']}] {item['text']} (page {item['page']})")
    
    json_str = json.dumps(result, indent=4, ensure_ascii=False)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    print(f"✅ Output saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test2.py <pdf_path>")
        print("Example: python3 test2.py ./training_pdfs/1-page.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"🔍 Processing: {pdf_path}")

    # Extract features
    features = extract_features_from_pdf(pdf_path)
    if not features:
        print("❌ No extractable text spans found.")
        sys.exit(1)

    print(f"📝 Extracted {len(features)} text spans")
    df = pd.DataFrame(features)
    
    # Debug: Show sample data
    print("\n📊 Sample extracted features:")
    print(df[['text', 'font_size', 'bold', 'italic', 'page']].head())

    # Try to load model and predict
    try:
        model, reverse_map, expected_features = load_model_and_mapping()
        df_pred = predict_headings(df, model, reverse_map, expected_features)
        
        # Check if model is working (not all predictions are the same)
        unique_predictions = df_pred['predicted_label'].nunique()
        if unique_predictions == 1 and len(df_pred) > 5:
            print("\n⚠️  Model predicting only one class, falling back to heuristics...")
            df_pred = simple_heuristic_classification(df)
            
    except Exception as e:
        print(f"⚠️  Model prediction failed ({e}), using heuristic classification...")
        df_pred = simple_heuristic_classification(df)

    print("\n🧠 Final Predictions:")
    for _, row in df_pred.iterrows():
        print(f"[{row['predicted_label']:6}] {row['text']}")

    save_json(df_pred, pdf_path)