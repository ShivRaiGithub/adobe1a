import os
import sys
import json
import csv
import fitz  # PyMuPDF
import pandas as pd
from joblib import load
import numpy as np

MODEL_PATH = "model/heading_model.joblib"
LABEL_MAP_PATH = "model/label_map.json"
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
CSV_OUTPUT_DIR = "/app/output"  # Directory to save CSV files

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

def is_likely_header_footer(text, page_num):
    """
    Check if text is likely a header or footer based on content patterns
    """
    text_lower = text.lower().strip()
    
    # Common header/footer patterns
    header_footer_patterns = [
        # Page numbers
        r'^\d+$',  # Just a number
        r'^page\s*\d+',  # "Page 1", "Page 2", etc.
        r'^\d+\s*of\s*\d+$',  # "1 of 5", etc.
        r'^\d+\s*/\s*\d+$',  # "1/5", etc.
        
        # Common footer text
        r'^copyright',
        r'^©',
        r'^confidential',
        r'^proprietary',
        r'^draft',
        r'^version\s*\d',
        r'^rev\s*\d',
        r'^document\s*#',
        r'^doc\s*#',
        
        # Date patterns
        r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # MM/DD/YYYY or similar
        r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',    # YYYY/MM/DD or similar
        
        # URL-like patterns
        r'^www\.',
        r'^http',
        r'\.com$',
        r'\.org$',
        r'\.net$',
        
        # Header continuation patterns
        r'^continued',
        r'^cont\.',
    ]
    
    import re
    for pattern in header_footer_patterns:
        if re.match(pattern, text_lower):
            return True
    
    # Check for very short text that might be page numbers
    if len(text.strip()) <= 3 and text.strip().isdigit():
        return True
    
    # Check for repeated text across pages (common in headers/footers)
    # This would require storing previous page texts, but for now we'll skip this
    
    return False

def extract_features_from_pdf(pdf_path, filter_headers_footers=True, header_threshold=0.1, footer_threshold=0.9):
    """
    Extract features from PDF for both CSV generation and classification
    
    Args:
        pdf_path: Path to the PDF file
        filter_headers_footers: Whether to filter out headers and footers (default: True)
        header_threshold: Percentage of page height considered header zone (default: 0.1 = 10%)
        footer_threshold: Percentage of page height considered footer zone (default: 0.9 = 90%)
    """
    doc = fitz.open(pdf_path)
    data = []

    for page_num, page in enumerate(doc):
        # Get page dimensions for header/footer detection
        page_rect = page.rect
        page_height = page_rect.height
        page_width = page_rect.width
        
        # Define header and footer zones
        header_zone = page_height * header_threshold
        footer_zone = page_height * footer_threshold
        
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text or len(text) > 50 or len(text) < 3:  # Ignore text > 50 chars or < 3 chars
                        continue

                    y_position = span["bbox"][1]  # y0 coordinate
                    
                    # Skip potential headers and footers based on position (if filtering enabled)
                    if filter_headers_footers and (y_position <= header_zone or y_position >= footer_zone):
                        # Additional checks for common header/footer patterns
                        if is_likely_header_footer(text, page_num):
                            print(f"🚫 Filtered header/footer: '{text}' (page {page_num + 1}, y={y_position:.1f})")
                            continue
                    
                    # Calculate relative position on page (useful for model)
                    relative_y = y_position / page_height
                    relative_x = span["bbox"][0] / page_width
                    
                    features = {
                        "text": text,
                        "font_size": span.get("size", 0),
                        "bold": int("bold" in span.get("font", "").lower()),
                        "italic": int("italic" in span.get("font", "").lower() or "oblique" in span.get("font", "").lower()),
                        "x0": span["bbox"][0],
                        "y0": span["bbox"][1],
                        "x1": span["bbox"][2],
                        "y1": span["bbox"][3],
                        "page": page_num,
                        "is_near_top": int(relative_y <= 0.15),      # Near top of page
                        "is_near_bottom": int(relative_y >= 0.85),   # Near bottom of page
                        "is_near_edge": int(relative_x <= 0.1 or relative_x >= 0.9)  # Near left/right edge
                    }
                    data.append(features)
    
    doc.close()
    return data

def save_to_csv(data, pdf_path):
    """Save extracted features to CSV file, similar to generate_csv.py"""
    if not data:
        print("⚠️ No data to save to CSV")
        return None
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    csv_path = os.path.join(CSV_OUTPUT_DIR, f"{base_name}.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    
    print(f"📊 CSV saved to: {csv_path}")
    return csv_path

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
    
    # Header/footer detection features (if they exist from extraction)
    if "is_near_top" not in df.columns:
        df["is_near_top"] = 0
    if "is_near_bottom" not in df.columns:
        df["is_near_bottom"] = 0
    if "is_near_edge" not in df.columns:
        df["is_near_edge"] = 0
    
    return df

def filter_repeated_headings(df, min_pages_to_consider_repeat=2, max_pages_for_repeat=0.7):
    """
    Remove headings that are likely headers/footers based on position and patterns
    
    Args:
        df: DataFrame with predictions
        min_pages_to_consider_repeat: Minimum pages a text must appear on to be considered repeating
        max_pages_for_repeat: Maximum fraction of total pages a text can appear on (0.7 = 70%)
    """
    if len(df) == 0:
        return df
    
    total_pages = df['page'].nunique()
    print(f"\n🔍 Checking for header/footer patterns across {total_pages} pages...")
    
    # Get only heading predictions for analysis
    heading_labels = {"TITLE", "H1", "H2", "H3", "H4", "H5"}
    headings_df = df[df['predicted_label'].isin(heading_labels)].copy()
    
    if len(headings_df) == 0:
        print("✅ No headings found to analyze")
        return df
    
    # Check for headings that appear in same relative position across pages (likely headers/footers)
    position_repeats = set()
    for text in headings_df['text'].unique():
        text_instances = headings_df[headings_df['text'] == text]
        if len(text_instances) >= min_pages_to_consider_repeat:
            # Check if they appear consistently in header/footer zones
            near_top_ratio = text_instances['is_near_top'].mean()
            near_bottom_ratio = text_instances['is_near_bottom'].mean()
            if near_top_ratio > 0.6 or near_bottom_ratio > 0.6:  # 60% of instances in header/footer zones
                position_repeats.add(text)
                print(f"📍 Found position-based header/footer: '{text}' (appears in header/footer zone)")
    
    # Check for numbered patterns that are likely headers/footers (e.g., "Page 1", "Page 2")
    import re
    pattern_repeats = set()
    
    # Group texts by pattern (replace numbers with placeholder)
    pattern_groups = {}
    for text in headings_df['text'].unique():
        # Create pattern by replacing digits with placeholder
        pattern = re.sub(r'\d+', '#NUM#', text.strip())
        # Also replace roman numerals
        pattern = re.sub(r'\b[ivxlcdm]+\b', '#ROM#', pattern, flags=re.IGNORECASE)
        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(text)
    
    # Find patterns that appear on multiple pages and are likely headers/footers
    for pattern, texts in pattern_groups.items():
        if len(texts) >= min_pages_to_consider_repeat and ('#NUM#' in pattern or '#ROM#' in pattern):
            # Check if these texts appear across multiple pages
            pages_with_pattern = set()
            for text in texts:
                pages_with_pattern.update(headings_df[headings_df['text'] == text]['page'].unique())
            
            # Only consider it header/footer if it appears across pages AND mostly in header/footer positions
            if len(pages_with_pattern) >= min_pages_to_consider_repeat:
                # Check if most instances are in header/footer zones
                pattern_instances = headings_df[headings_df['text'].isin(texts)]
                header_footer_ratio = (pattern_instances['is_near_top'].sum() + pattern_instances['is_near_bottom'].sum()) / len(pattern_instances)
                
                if header_footer_ratio > 0.5:  # 50% in header/footer zones
                    pattern_repeats.update(texts)
                    print(f"📋 Found header/footer pattern '{pattern}': {texts[:3]}{'...' if len(texts) > 3 else ''}")
    
    # Combine header/footer detections
    all_header_footer_texts = position_repeats | pattern_repeats
    
    if position_repeats:
        print(f"🔍 Found {len(position_repeats)} position-based header/footer texts")
    
    if pattern_repeats:
        print(f"🔍 Found {len(pattern_repeats)} pattern-based header/footer texts")
    
    # Filter out header/footer texts from headings, but keep them as 'normal'
    filtered_count = 0
    
    for idx, row in df.iterrows():
        if row['text'] in all_header_footer_texts and row['predicted_label'] in heading_labels:
            original_label = row['predicted_label']
            df.loc[idx, 'predicted_label'] = 'normal'
            print(f"🔄 Changed '{row['text']}' from {original_label} to normal (header/footer)")
            filtered_count += 1
    
    if filtered_count > 0:
        print(f"✅ Filtered out {filtered_count} header/footer headings")
    else:
        print("✅ No header/footer headings found to filter")
    
    return df

def final_heading_cleanup(df):
    """
    Final cleanup of heading list to remove excessive over-repetition
    """
    print("\n🧹 Final heading cleanup...")
    
    heading_labels = {"TITLE", "H1", "H2", "H3", "H4", "H5"}
    headings_df = df[df['predicted_label'].isin(heading_labels)].copy()
    
    if len(headings_df) == 0:
        return df
    
    # Count frequency of each heading text
    heading_counts = headings_df['text'].value_counts()
    
    # Find headings that appear excessively frequently (more than half the total pages)
    total_pages = df['page'].nunique()
    max_allowed_occurrences = max(3, total_pages // 2)  # At most half of total pages, minimum 3
    frequent_headings = heading_counts[heading_counts > max_allowed_occurrences].index.tolist()
    
    if frequent_headings:
        print(f"🚨 Found excessively frequent headings (>{max_allowed_occurrences} times):")
        for heading in frequent_headings:
            count = heading_counts[heading]
            print(f"   '{heading}' appears {count} times")
    
    # Keep only reasonable number of occurrences of excessively repeated headings
    filtered_count = 0
    heading_occurrence_count = {}
    
    for idx, row in df.iterrows():
        if row['predicted_label'] in heading_labels and row['text'] in frequent_headings:
            if row['text'] not in heading_occurrence_count:
                heading_occurrence_count[row['text']] = 0
            
            heading_occurrence_count[row['text']] += 1
            
            # Keep reasonable number of occurrences, filter out excessive ones
            if heading_occurrence_count[row['text']] > max_allowed_occurrences:
                original_label = row['predicted_label']
                df.loc[idx, 'predicted_label'] = 'normal'
                print(f"🗑️  Removed excess occurrence of '{row['text']}' (#{heading_occurrence_count[row['text']]}) from {original_label}")
                filtered_count += 1
    
    if filtered_count > 0:
        print(f"✅ Removed {filtered_count} excessive heading occurrences")
    else:
        print("✅ No excessive heading occurrences found")
    
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

    # Always add enhanced features since the model was trained with them
    print("🔧 Adding enhanced features...")
    df = enhance_features(df)
    
    # Use the expected features from training
    if expected_features is not None:
        feature_cols = expected_features
        print(f"🔧 Using saved feature list: {len(feature_cols)} features")
    else:
        # Default to all enhanced features
        feature_cols = [
            "font_size", "bold", "italic", "x", "y", "page",
            "text_length", "word_count", "is_uppercase", "is_titlecase",
            "ends_with_punctuation", "starts_with_number", "has_colon",
            "relative_x", "relative_y", "font_size_rank", "is_largest_font",
            "font_size_percentile", "has_digits", "has_special_chars", "starts_with_capital",
            "is_near_top", "is_near_bottom", "is_near_edge"
        ]
        print(f"🔧 Using default enhanced features: {len(feature_cols)} features")
    
    # Ensure proper data types
    type_mapping = {
        "font_size": float, "bold": int, "italic": int, "x": float, "y": float, "page": int,
        "text_length": int, "is_uppercase": int, "word_count": int, "ends_with_punctuation": int,
        "starts_with_number": int, "has_colon": int, "relative_x": float, "relative_y": float,
        "font_size_rank": float, "is_largest_font": int, "font_size_percentile": float,
        "has_digits": int, "is_titlecase": int, "has_special_chars": int, "starts_with_capital": int,
        "is_near_top": int, "is_near_bottom": int, "is_near_edge": int
    }
    
    for col in feature_cols:
        if col in df.columns and col in type_mapping:
            df[col] = df[col].astype(type_mapping[col])

    X = df[feature_cols]
    
    # Debug: Print feature statistics
    print(f"\n📊 Using {len(feature_cols)} features")
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

    # Filter out repeated headings that appear across multiple pages
    df = filter_repeated_headings(df)
    
    # Final cleanup to remove excessive duplicates
    df = final_heading_cleanup(df)

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
            ((first_page['text'].str.len() <= 50) & (first_page['text'].str.len() >= 3))  # Updated to match filtering criteria
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
            is_short = len(row['text']) <= 60 and len(row['text']) >= 3  # Updated to match filtering criteria
            is_sentence_case = not row['text'].islower()
            
            heading_score = sum([is_large_font, is_bold, is_short, is_sentence_case])
            
            if heading_score >= 2:
                if row['font_size'] >= df['font_size'].quantile(0.9):
                    df.loc[idx, 'predicted_label'] = 'H1'
                elif row['font_size'] >= df['font_size'].quantile(0.8):
                    df.loc[idx, 'predicted_label'] = 'H2'
                else:
                    df.loc[idx, 'predicted_label'] = 'H3'
    
    # Also filter repeated headings in heuristic classification
    df = filter_repeated_headings(df)
    
    # Final cleanup to remove excessive duplicates
    df = final_heading_cleanup(df)
    
    return df

def save_json(df, pdf_filename):
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
    heading_labels = {"H1", "H2", "H3", "H4", "H5"}  # Exclude "TITLE" from headings
    
    for _, row in df.iterrows():
        if row["predicted_label"] in heading_labels:
            outline.append({
                "level": row["predicted_label"],
                "text": row["text"].strip(),
                "page": int(row["page"]) + 1  # Convert to 1-based page numbering
            })

    result = {"title": title.strip(), "outline": outline}
    
    # Pretty print result
    print(f"\n📄 Document Analysis Results:")
    print(f"Title: {result['title']}")
    print(f"Headings found: {len(outline)}")
    for item in outline:
        print(f"  [{item['level']}] {item['text']} (page {item['page']})")
    
    json_str = json.dumps(result, indent=4, ensure_ascii=False)

    # Save JSON with same basename as PDF
    base_name = os.path.splitext(pdf_filename)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    print(f"✅ JSON output saved to: {out_path}")
    return out_path

def process_single_pdf(pdf_path):
    """Process a single PDF file"""
    pdf_filename = os.path.basename(pdf_path)
    print(f"\n🔍 Processing: {pdf_filename}")
    
    filter_headers_footers = True  # Always enable header/footer filtering
    print("🚫 Header/footer filtering: ENABLED")

    # Extract features
    features = extract_features_from_pdf(pdf_path, filter_headers_footers=filter_headers_footers)
    if not features:
        print("❌ No extractable text spans found.")
        return False

    print(f"📝 Extracted {len(features)} text spans")
    
    # Save to CSV (like generate_csv.py)
    csv_path = save_to_csv(features, pdf_path)
    
    # Create DataFrame for classification
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

    # Save JSON output with corresponding filename
    json_output_path = save_json(df_pred, pdf_filename)
    
    print(f"\n🎉 Processing completed for {pdf_filename}!")
    print(f"📊 CSV file: {csv_path}")
    print(f"📄 JSON output: {json_output_path}")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting PDF processing...")
    print(f"📁 Input directory: {INPUT_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    # Find all PDF files in input directory
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {INPUT_DIR}")
        sys.exit(1)
    
    print(f"📚 Found {len(pdf_files)} PDF file(s) to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    # Process each PDF file
    processed_count = 0
    failed_count = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        try:
            success = process_single_pdf(pdf_path)
            if success:
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"❌ Failed to process {pdf_file}: {e}")
            failed_count += 1
    
    print(f"\n🏁 Processing Summary:")
    print(f"✅ Successfully processed: {processed_count} files")
    print(f"❌ Failed to process: {failed_count} files")
    print(f"📁 Results saved in: {OUTPUT_DIR}")
    
    if failed_count > 0:
        sys.exit(1)  # Exit with error code if any files failed
    else:
        print("🎉 All files processed successfully!")