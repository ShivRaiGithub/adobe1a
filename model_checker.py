#!/usr/bin/env python3
import os
import json
from joblib import load
import pandas as pd

MODEL_PATH = "model/heading_model.joblib"
LABEL_MAP_PATH = "model/label_map.json"

def check_model():
    print("🔍 Checking model configuration...")
    
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"❌ Label map not found: {LABEL_MAP_PATH}")
        return
    
    try:
        # Load model
        model = load(MODEL_PATH)
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        
        # Check if model has feature_names_ attribute
        if hasattr(model, 'feature_names_in_'):
            print(f"   Expected features ({len(model.feature_names_in_)}): {list(model.feature_names_in_)}")
        elif hasattr(model, 'get_booster'):
            # For XGBoost
            booster = model.get_booster()
            if hasattr(booster, 'feature_names') and booster.feature_names:
                print(f"   Expected features ({len(booster.feature_names)}): {booster.feature_names}")
            else:
                print("   ⚠️  Feature names not stored in model")
        else:
            print("   ⚠️  Cannot determine expected features")
        
        # Load label mapping
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        
        print(f"✅ Label mapping loaded:")
        for label, code in label_map.items():
            print(f"   {label} → {code}")
        
        # Check for features.json
        features_path = "model/features.json"
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                saved_features = json.load(f)
            print(f"✅ Feature list found ({len(saved_features)} features):")
            for i, feature in enumerate(saved_features):
                print(f"   {i+1:2d}. {feature}")
        else:
            print("⚠️  No feature list file found (model/features.json)")
        
        # Test with basic features
        print("\n🧪 Testing model with basic features...")
        basic_features = ["font_size", "bold", "italic", "x", "y", "page"]
        test_data = pd.DataFrame({
            "font_size": [12.0],
            "bold": [0],
            "italic": [0], 
            "x": [100.0],
            "y": [200.0],
            "page": [0]
        })
        
        try:
            prediction = model.predict(test_data)
            print(f"✅ Basic features work - prediction: {prediction}")
        except Exception as e:
            print(f"❌ Basic features failed: {e}")
            
            # Try with enhanced features
            print("🧪 Testing with enhanced features...")
            enhanced_data = pd.DataFrame({
                "font_size": [12.0], "bold": [0], "italic": [0], "x": [100.0], "y": [200.0], "page": [0],
                "text_length": [10], "word_count": [2], "is_uppercase": [0], "is_titlecase": [1],
                "ends_with_punctuation": [0], "starts_with_number": [0], "has_colon": [0],
                "relative_x": [0.5], "relative_y": [0.5], "font_size_rank": [1.0], "is_largest_font": [1],
                "font_size_percentile": [0.8], "has_digits": [0], "has_special_chars": [0], "starts_with_capital": [1]
            })
            
            try:
                prediction = model.predict(enhanced_data)
                print(f"✅ Enhanced features work - prediction: {prediction}")
            except Exception as e2:
                print(f"❌ Enhanced features also failed: {e2}")
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    check_model()