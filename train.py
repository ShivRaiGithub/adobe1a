import os
import pandas as pd
import json
import sys
import csv
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from joblib import dump
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix
from difflib import SequenceMatcher
import re
import numpy as np
from collections import Counter
import fitz  # PyMuPDF
from tqdm import tqdm

TRAIN_CSV_DIR = "training_csvs"
TRAIN_PDF_DIR = "training_pdfs"
TRAIN_JSON_DIR = "training_jsons"
MODEL_PATH = "model/heading_model.joblib"

os.makedirs("model", exist_ok=True)
os.makedirs(TRAIN_CSV_DIR, exist_ok=True)

def extract_features_from_pdf(pdf_path):
    """Extract text features from PDF file"""
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
    doc.close()
    return data

def save_to_csv(data, output_csv_path):
    """Save extracted features to CSV file"""
    if not data:
        return
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)

def convert_pdfs_to_csvs(pdf_dir, csv_dir):
    """Convert all PDFs in directory to CSV files"""
    os.makedirs(csv_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    print(f"🔄 Converting {len(pdf_files)} PDF files to CSV...")
    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        pdf_path = os.path.join(pdf_dir, pdf_file)
        csv_filename = os.path.splitext(pdf_file)[0] + ".csv"
        csv_path = os.path.join(csv_dir, csv_filename)

        # Skip if CSV already exists and is newer than PDF
        if os.path.exists(csv_path) and os.path.getmtime(csv_path) > os.path.getmtime(pdf_path):
            continue

        try:
            data = extract_features_from_pdf(pdf_path)
            save_to_csv(data, csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to convert {pdf_file}: {e}")

def normalize(text):
    return re.sub(r'\W+', '', text.lower().strip())

def is_match(text1, text2, threshold=0.8):
    return SequenceMatcher(None, normalize(text1), normalize(text2)).ratio() >= threshold

def get_label_for_row(text, page, title, outline, title_assigned):
    if title and not title_assigned and is_match(text, title, threshold=0.7):
        return "TITLE"
    for item in outline:
        if item["page"] == page and is_match(text, item["text"], threshold=0.7):
            return item["level"]
    return "normal"

def enhance_features(df):
    """Add additional features to improve classification"""
    
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

def balance_dataset(df, target_samples_per_class=1000):
    """Balance the dataset by undersampling majority class and oversampling minority classes"""
    
    print("📊 Original class distribution:")
    print(df["label"].value_counts())
    
    balanced_dfs = []
    
    for label in df["label"].unique():
        label_df = df[df["label"] == label]
        current_count = len(label_df)
        
        if current_count > target_samples_per_class:
            # Undersample
            label_df = label_df.sample(n=target_samples_per_class, random_state=42)
            print(f"🔽 {label}: {current_count} → {target_samples_per_class} (undersampled)")
        elif current_count < target_samples_per_class:
            # Oversample with replacement
            label_df = label_df.sample(n=target_samples_per_class, replace=True, random_state=42)
            print(f"🔼 {label}: {current_count} → {target_samples_per_class} (oversampled)")
        else:
            print(f"➡️  {label}: {current_count} (no change)")
            
        balanced_dfs.append(label_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print("\n📊 Balanced class distribution:")
    print(balanced_df["label"].value_counts())
    
    return balanced_df

def main():
    """Main training function"""
    # Convert PDFs to CSVs if PDF directory exists
    if os.path.exists(TRAIN_PDF_DIR) and os.listdir(TRAIN_PDF_DIR):
        convert_pdfs_to_csvs(TRAIN_PDF_DIR, TRAIN_CSV_DIR)

    rows = []
    file_count = 0
    print("📦 Processing training files...")

    for file in os.listdir(TRAIN_CSV_DIR):
        if file.endswith(".csv"):
            base = os.path.splitext(file)[0]
            csv_path = os.path.join(TRAIN_CSV_DIR, file)
            json_path = os.path.join(TRAIN_JSON_DIR, base + ".json")

            if not os.path.exists(json_path):
                print(f"[WARN] JSON not found for {file}, skipping.")
                continue

            try:
                df = pd.read_csv(csv_path)
                with open(json_path, "r", encoding='utf-8') as f:
                    js = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")
                continue

            title = js.get("title", "") or ""
            outline = js.get("outline", []) or []

            title_assigned = False
            file_rows = 0

            for _, row in df.iterrows():
                text = str(row["text"]).strip()
                if len(text) > 80 or len(text) < 2:  # Skip very short and very long text
                    continue

                label = get_label_for_row(text, int(row["page"]), title, outline, title_assigned)

                if label == "TITLE":
                    title_assigned = True

                rows.append({
                    "text": text,
                    "font_size": float(row["font_size"]),
                    "bold": int(row["bold"]),
                    "italic": int(row["italic"]),
                    "x": float(row["x0"]),
                    "y": float(row["y0"]),
                    "page": int(row["page"]),
                    "label": label
                })
                file_rows += 1
            
            file_count += 1
            print(f"✅ Processed {file}: {file_rows} rows")

    print(f"\n📊 Total files processed: {file_count}")
    print(f"📊 Total training rows: {len(rows)}")

    if len(rows) == 0:
        print("❌ No training data found!")
        sys.exit(1)

    df_all = pd.DataFrame(rows)

    # Enhance features
    print("🔧 Engineering features...")
    df_all = enhance_features(df_all)

    print("📊 Original class distribution:")
    print(df_all["label"].value_counts())

    # Balance dataset
    df_all = balance_dataset(df_all, target_samples_per_class=500)

    # Encode labels
    le = LabelEncoder()
    df_all["label_enc"] = le.fit_transform(df_all["label"])

    # Save label mapping
    le_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    with open("model/label_map.json", "w") as f:
        json.dump(le_mapping, f, indent=2)

    print(f"🗺️ Label mapping: {le_mapping}")

    # Select features for training
    feature_cols = [
        "font_size", "bold", "italic", "x", "y", "page",
        "text_length", "word_count", "is_uppercase", "is_titlecase",
        "ends_with_punctuation", "starts_with_number", "has_colon",
        "relative_x", "relative_y", "font_size_rank", "is_largest_font",
        "font_size_percentile", "has_digits", "has_special_chars", "starts_with_capital"
    ]

    X = df_all[feature_cols]
    y = df_all["label_enc"]

    print(f"🔧 Training with {len(feature_cols)} features: {feature_cols}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model with better parameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    print("🎯 Training model...")
    model.fit(X_train, y_train)

    # Evaluate model
    print("\n📊 Model Evaluation:")
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Validation accuracy: {val_score:.3f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Detailed classification report
    y_pred = model.predict(X_val)
    print("\n📋 Classification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔍 Top 10 Feature Importances:")
    print(feature_importance.head(10))

    # Save model
    dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

    # Save feature list for reference
    with open("model/features.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\n🎉 Training completed successfully!")
    print(f"📁 Model files saved in 'model/' directory")
    print(f"📊 Final dataset size: {len(df_all)} samples")
    print(f"🎯 Validation accuracy: {val_score:.3f}")

# Convert PDFs to CSVs if PDF directory exists
if os.path.exists(TRAIN_PDF_DIR) and os.listdir(TRAIN_PDF_DIR):
    convert_pdfs_to_csvs(TRAIN_PDF_DIR, TRAIN_CSV_DIR)

rows = []
file_count = 0
print("📦 Processing training files...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heading classification model from PDFs or CSVs.")
    parser.add_argument("--pdf_dir", type=str, help="Path to directory with training PDFs (optional, defaults to training_pdfs)")
    parser.add_argument("--json_dir", type=str, help="Path to directory with training JSONs (optional, defaults to training_jsons)")
    parser.add_argument("--csv_dir", type=str, help="Path to directory with training CSVs (optional, defaults to training_csvs)")
    parser.add_argument("--convert_only", action="store_true", help="Only convert PDFs to CSVs, don't train model")
    
    args = parser.parse_args()
    
    # Update directories if provided
    if args.pdf_dir:
        TRAIN_PDF_DIR = args.pdf_dir
    if args.json_dir:
        TRAIN_JSON_DIR = args.json_dir  
    if args.csv_dir:
        TRAIN_CSV_DIR = args.csv_dir
    
    # If convert_only flag is set, just convert PDFs and exit
    if args.convert_only:
        if os.path.exists(TRAIN_PDF_DIR) and os.listdir(TRAIN_PDF_DIR):
            convert_pdfs_to_csvs(TRAIN_PDF_DIR, TRAIN_CSV_DIR)
            print("✅ PDF to CSV conversion completed!")
        else:
            print("❌ No PDFs found in the specified directory!")
        sys.exit(0)
    
    # Run main training
    main()