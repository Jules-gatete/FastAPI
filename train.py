"""
Train ML models for pharmaceutical disposal prediction (IMPROVED VERSION).
Predicts: Dosage Form, Manufacturer, Disposal Category, Method of Disposal,
         Handling Method, and Disposal Remarks from Generic Name.

IMPROVEMENTS:
- Normalizes text to lowercase to eliminate case sensitivity issues
- Groups similar Dosage Forms (standardizes variations)
- Merges rare classes (<3 occurrences) into "OTHER" category
- Uses XGBoost for better performance on imbalanced data
- Adds comprehensive data preprocessing
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback to RandomForest if not available
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not available. Using RandomForest instead.")
    USE_XGBOOST = False

# Configuration
DATA_FILE = 'rwanda_fda_medicines_with_disposal.csv'
MODELS_DIR = 'models'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
MIN_CLASS_FREQUENCY = 3  # Minimum occurrences to keep a class, else merge to "OTHER"

# Output columns
OUTPUT_COLUMNS = {
    'dosage_form': 'Dosage Form',
    'manufacturer': "Manufacturer's Name",
    'disposal_category': 'Disposal Category',
    'method_of_disposal': 'Method of Disposal',
    'handling_method': 'Handling Method',
    'disposal_remarks': 'Disposal Remarks'
}

def create_models_directory():
    """Create models directory if it doesn't exist."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

def normalize_text(text):
    """Normalize text: lowercase, strip, remove extra spaces."""
    if pd.isna(text):
        return ''
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

def standardize_dosage_form(df):
    """
    Standardize Dosage Form variations.
    Maps common variations to standard forms.
    """
    print("\nStandardizing Dosage Forms...")
    
    # Create mapping for common variations
    dosage_mapping = {
        # Tablets variations
        'tablet': 'tablets',
        'tablets': 'tablets',
        'film coated tablet': 'film coated tablets',
        'film-coated tablet': 'film coated tablets',
        'film coated tablets': 'film coated tablets',
        'film-coated tablets': 'film coated tablets',
        'coated tablet': 'coated tablets',
        'coated tablets': 'coated tablets',
        'uncoated tablet': 'uncoated tablets',
        'uncoated tablets': 'uncoated tablets',
        'effervescent tablet': 'effervescent tablets',
        'effervescent tablets': 'effervescent tablets',
        'effervescents tablets': 'effervescent tablets',
        'efervescent secable tablets': 'effervescent tablets',
        'sustained release tablet': 'sustained release tablets',
        'sustained release tablets': 'sustained release tablets',
        
        # Capsules variations
        'capsule': 'capsules',
        'capsules': 'capsules',
        'hard gelatin capsule': 'hard gelatin capsules',
        'hard gelatin capsules': 'hard gelatin capsules',
        'caplet': 'caplets',
        'caplets': 'caplets',
        
        # Solutions variations
        'solution for injection': 'solution for injection',
        'solution for infusion': 'solution for infusion',
        'solution for intravenous infusion': 'solution for infusion',
        'solution for iv infusion': 'solution for infusion',
        'intravenous solution for infusion': 'solution for infusion',
        'intravenous infusion': 'solution for infusion',
        
        # Suspensions variations
        'suspension': 'suspension',
        'oral suspension': 'suspension',
        'syrup': 'syrup',
        
        # Powder variations
        'powder for oral solution': 'powder for oral suspension',
        'powder for oral suspension': 'powder for oral suspension',
        'powder for injection': 'powder for injection',
        'granules': 'granules',
        'granular powder': 'granules',
        'granulated oral powder': 'granules',
        'crystalline powder': 'powder',
        'powder': 'powder',
        
        # Other variations
        'suppository': 'suppositories',
        'suppositories': 'suppositories',
        'cream': 'cream',
        'ointment': 'ointment',
        'shampoo': 'shampoo',
        'injection': 'injection',
        'injectable': 'injection',
    }
    
    # Normalize all dosage forms
    df['Dosage Form Normalized'] = df['Dosage Form'].apply(normalize_text)
    
    # Apply mapping
    df['Dosage Form Standardized'] = df['Dosage Form Normalized'].map(
        dosage_mapping
    ).fillna(df['Dosage Form Normalized'])
    
    # Count before and after
    before_count = df['Dosage Form'].nunique()
    after_count = df['Dosage Form Standardized'].nunique()
    print(f"  Before: {before_count} unique forms")
    print(f"  After standardization: {after_count} unique forms")
    print(f"  Reduced by: {before_count - after_count} forms")
    
    return df

def merge_rare_classes(series, min_frequency=MIN_CLASS_FREQUENCY, other_label='OTHER'):
    """
    Merge rare classes (with <min_frequency occurrences) into 'OTHER' category.
    Returns the modified series and a mapping dictionary.
    """
    counts = series.value_counts()
    rare_classes = counts[counts < min_frequency].index.tolist()
    
    if len(rare_classes) > 0:
        print(f"  Merging {len(rare_classes)} rare classes into '{other_label}'")
        print(f"  Rare classes: {rare_classes[:10]}{'...' if len(rare_classes) > 10 else ''}")
        
        # Create mapping
        mapping = {cls: other_label if cls in rare_classes else cls for cls in series.unique()}
        
        # Apply mapping
        series_modified = series.map(mapping)
        
        return series_modified, mapping
    else:
        print(f"  No rare classes found (all classes have >= {min_frequency} occurrences)")
        return series, {}

def load_and_preprocess_data():
    """Load and preprocess the dataset with comprehensive cleaning."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} records")
    
    # Handle missing Generic Names
    initial_count = len(df)
    df = df.dropna(subset=['Generic Name'])
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} records with missing Generic Names")
    
    # Normalize Generic Name
    df['Generic Name'] = df['Generic Name'].apply(normalize_text)
    df = df[df['Generic Name'] != '']
    
    # Standardize Dosage Form
    df = standardize_dosage_form(df)
    
    # Normalize other text columns
    df['Manufacturer Normalized'] = df["Manufacturer's Name"].apply(normalize_text)
    df['Disposal Category Normalized'] = df['Disposal Category'].apply(normalize_text)
    
    print(f"Final dataset size: {len(df)} records")
    
    return df

def extract_features(generic_names):
    """
    Extract engineered features from Generic Names.
    Returns a feature matrix with engineered features.
    """
    features = []
    
    for name in generic_names:
        name_lower = name.lower()
        name_features = []
        
        # 1. Length features
        name_features.append(len(name))
        name_features.append(len(name.split()))
        name_features.append(len(name.split(',')))  # Number of compounds
        
        # 2. Keyword features (common pharmaceutical terms)
        keywords = {
            'tablet': 'tablet' in name_lower,
            'capsule': 'capsule' in name_lower,
            'injection': 'injection' in name_lower or 'injectable' in name_lower,
            'suspension': 'suspension' in name_lower,
            'syrup': 'syrup' in name_lower,
            'solution': 'solution' in name_lower,
            'cream': 'cream' in name_lower,
            'ointment': 'ointment' in name_lower,
            'powder': 'powder' in name_lower,
            'liquid': 'liquid' in name_lower,
            'solid': 'solid' in name_lower,
            'hcl': 'hcl' in name_lower or 'hydrochloride' in name_lower,
            'sodium': 'sodium' in name_lower,
            'calcium': 'calcium' in name_lower,
            'sulfate': 'sulfate' in name_lower or 'sulphate' in name_lower,
            'acetate': 'acetate' in name_lower,
            'fumarate': 'fumarate' in name_lower,
            'phosphate': 'phosphate' in name_lower,
            'mg': 'mg' in name_lower or 'milligram' in name_lower,
            'ml': 'ml' in name_lower or 'milliliter' in name_lower,
            'compound': ',' in name or 'and' in name_lower,
        }
        
        # Add keyword features as binary
        name_features.extend([1 if keywords[k] else 0 for k in sorted(keywords.keys())])
        
        # 3. Dosage strength patterns
        # Extract numbers (likely dosage strengths)
        numbers = re.findall(r'\d+', name)
        name_features.append(len(numbers))  # Number of numeric values
        if numbers:
            name_features.append(max([int(n) for n in numbers]))  # Max number
            name_features.append(min([int(n) for n in numbers]))  # Min number
        else:
            name_features.extend([0, 0])
        
        # 4. Character features
        name_features.append(name.count(' '))
        name_features.append(name.count(','))
        name_features.append(name.count('/'))
        name_features.append(name.count('-'))
        
        # 5. Common prefixes/suffixes
        prefixes = {
            'anti': name_lower.startswith('anti'),
            'co': name_lower.startswith('co'),
            'de': name_lower.startswith('de'),
            'pre': name_lower.startswith('pre'),
            'pro': name_lower.startswith('pro'),
        }
        name_features.extend([1 if prefixes[k] else 0 for k in sorted(prefixes.keys())])
        
        features.append(name_features)
    
    return np.array(features)

def create_embeddings(df, embedding_model):
    """Create sentence embeddings for Generic Names."""
    print("\nCreating sentence embeddings...")
    generic_names = df['Generic Name'].tolist()
    embeddings = embedding_model.encode(
        generic_names,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"Created embeddings of shape: {embeddings.shape}")
    return embeddings

def create_combined_features(embeddings, engineered_features):
    """Combine embeddings with engineered features."""
    print(f"\nCombining embeddings ({embeddings.shape[1]} dims) with engineered features ({engineered_features.shape[1]} dims)...")
    combined = np.hstack([embeddings, engineered_features])
    print(f"Combined features shape: {combined.shape}")
    return combined

def create_broad_categories(y_dosage, y_manufacturer):
    """
    Create broad categories for two-stage approach.
    Returns broad categories for dosage forms and manufacturers.
    """
    # Dosage Form broad categories
    dosage_broad = []
    for form in y_dosage:
        form_lower = str(form).lower()
        if 'tablet' in form_lower:
            dosage_broad.append('tablet_category')
        elif 'capsule' in form_lower:
            dosage_broad.append('capsule_category')
        elif 'injection' in form_lower or 'injectable' in form_lower or 'solution for injection' in form_lower:
            dosage_broad.append('injection_category')
        elif 'suspension' in form_lower or 'syrup' in form_lower:
            dosage_broad.append('liquid_category')
        elif 'powder' in form_lower or 'granule' in form_lower:
            dosage_broad.append('powder_category')
        elif 'cream' in form_lower or 'ointment' in form_lower:
            dosage_broad.append('topical_category')
        elif 'suppository' in form_lower:
            dosage_broad.append('suppository_category')
        else:
            dosage_broad.append('other_category')
    
    # Manufacturer broad categories (based on common patterns)
    manufacturer_broad = []
    for mfg in y_manufacturer:
        mfg_lower = str(mfg).lower()
        # Common manufacturer patterns
        if 'laboratories' in mfg_lower or 'lab' in mfg_lower:
            manufacturer_broad.append('laboratory_category')
        elif 'pharma' in mfg_lower or 'pharmaceutical' in mfg_lower:
            manufacturer_broad.append('pharma_category')
        elif 'limited' in mfg_lower or 'ltd' in mfg_lower:
            manufacturer_broad.append('limited_category')
        elif 'sas' in mfg_lower or 'sa' in mfg_lower:
            manufacturer_broad.append('sas_category')
        else:
            manufacturer_broad.append('other_category')
    
    return np.array(dosage_broad), np.array(manufacturer_broad)

def train_two_stage_model(X, y_broad, y_specific, model_name, 
                          n_estimators_stage1=100, n_estimators_stage2=200, 
                          use_xgboost=USE_XGBOOST):
    """
    Train a two-stage model:
    Stage 1: Predict broad category
    Stage 2: Predict specific class within that category
    """
    print(f"\nTraining {model_name} two-stage classifier...")
    print(f"  Broad categories: {len(np.unique(y_broad))}")
    print(f"  Specific classes: {len(np.unique(y_specific))}")
    

    # Stage 1: Predict broad category
    print(f"\n  Stage 1: Training broad category classifier...")

    # If using XGBoost, it expects numeric class labels for multiclass; encode them
    stage1_label_encoder = None
    y_broad_for_fit = y_broad
    if use_xgboost:
        stage1_label_encoder = LabelEncoder()
        y_broad_for_fit = stage1_label_encoder.fit_transform(y_broad)

    if use_xgboost:
        stage1_model = XGBClassifier(
            n_estimators=n_estimators_stage1,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    else:
        stage1_model = RandomForestClassifier(
            n_estimators=n_estimators_stage1,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    stage1_model.fit(X, y_broad_for_fit)
    stage1_score = stage1_model.score(X, y_broad_for_fit)
    print(f"    Stage 1 training accuracy: {stage1_score:.4f}")
    
    # Stage 2: Predict specific class within each category
    print(f"\n  Stage 2: Training specific class classifiers...")
    stage2_models = {}
    stage2_label_encoders = {}
    
    for category in np.unique(y_broad):
        # Get indices for this category
        category_mask = y_broad == category
        if np.sum(category_mask) < 2:  # Skip if too few samples
            continue
        
        X_category = X[category_mask]
        y_category = y_specific[category_mask]
        
        if len(np.unique(y_category)) < 2:  # Skip if only one class
            continue
        
        print(f"    Training {category} classifier ({np.sum(category_mask)} samples, {len(np.unique(y_category))} classes)...")
        
        # Prepare label encoding for XGBoost which expects numeric class labels
        y_category_for_fit = y_category
        encoder = None
        if use_xgboost:
            encoder = LabelEncoder()
            try:
                y_category_for_fit = encoder.fit_transform(y_category)
            except Exception:
                # Fallback to keeping original labels if encoding fails
                y_category_for_fit = y_category

        if use_xgboost:
            stage2_model = XGBClassifier(
                n_estimators=n_estimators_stage2,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        else:
            stage2_model = RandomForestClassifier(
                n_estimators=n_estimators_stage2,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        stage2_model.fit(X_category, y_category_for_fit)
        stage2_score = stage2_model.score(X_category, y_category_for_fit)
        print(f"      {category} training accuracy: {stage2_score:.4f}")

        stage2_models[category] = stage2_model
        if encoder is not None:
            stage2_label_encoders[category] = encoder
    
    # Return the stage1 model, optional encoder, and stage2 dict
    # include stage2 label encoders (may be empty) for decoding at inference
    return {
        'stage1': stage1_model,
        'stage1_le': stage1_label_encoder,
        'stage2': stage2_models,
        'stage2_le': stage2_label_encoders
    }

def train_categorical_model(X, y, model_name, n_estimators=200, use_xgboost=USE_XGBOOST):
    """Train classifier for categorical outputs (XGBoost or RandomForest)."""
    print(f"\nTraining {model_name} classifier...")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Samples: {len(y)}")
    
    # Calculate class distribution
    class_counts = Counter(y)
    print(f"  Most common class: {class_counts.most_common(1)[0][1]} occurrences")
    print(f"  Least common class: {min(class_counts.values())} occurrences")
    
    # Prepare label encoding when using XGBoost (it expects numeric labels)
    label_encoder = None
    y_for_fit = y
    if use_xgboost:
        try:
            label_encoder = LabelEncoder()
            y_for_fit = label_encoder.fit_transform(y)
            print("  Encoded categorical labels for XGBoost")
        except Exception:
            # If encoding fails, fall back to original labels
            y_for_fit = y

    if use_xgboost:
        # Use XGBoost with scale_pos_weight for class imbalance
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        print("  Using XGBoost classifier")
    else:
        # Fallback to RandomForest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        print("  Using RandomForest classifier")

    model.fit(X, y_for_fit)

    # Calculate accuracy on training set
    try:
        train_score = model.score(X, y_for_fit)
        print(f"  Training accuracy: {train_score:.4f}")
    except Exception:
        pass

    # Return model and encoder (encoder may be None)
    return model, label_encoder

def prepare_multilabel_data(df):
    """Prepare multi-label data for Method of Disposal."""
    methods = df['Method of Disposal'].fillna('').astype(str)
    # Split comma-separated values and normalize
    method_lists = [m.split(',') if m else [] for m in methods]
    method_lists = [[normalize_text(item) for item in lst] for lst in method_lists]
    
    # Create MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_multilabel = mlb.fit_transform(method_lists)
    
    print(f"\nMulti-label Method of Disposal:")
    print(f"  Total unique labels: {len(mlb.classes_)}")
    print(f"  Labels: {list(mlb.classes_)}")
    
    return y_multilabel, mlb

def train_multilabel_model(X, y_multilabel, mlb, use_xgboost=USE_XGBOOST):
    """Train multi-label classifier for Method of Disposal."""
    print("\nTraining Method of Disposal multi-label classifier...")
    
    if use_xgboost:
        base_model = XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        print("  Using XGBoost for multi-label classification")
    else:
        base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        print("  Using RandomForest for multi-label classification")
    
    multilabel_model = MultiOutputClassifier(base_model)
    multilabel_model.fit(X, y_multilabel)
    
    # Calculate accuracy
    train_predictions = multilabel_model.predict(X)
    accuracy = np.mean(train_predictions == y_multilabel)
    print(f"  Training accuracy: {accuracy:.4f}")
    
    return multilabel_model, mlb

def create_similarity_index(df, embeddings):
    """Create similarity search index for text retrieval."""
    print("\nCreating similarity search index for text outputs...")
    
    # Create NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
    nn_model.fit(embeddings)
    
    # Store the data for retrieval
    similarity_data = {
        'embeddings': embeddings,
        'handling_methods': df['Handling Method'].values,
        'disposal_remarks': df['Disposal Remarks'].values,
        'generic_names': df['Generic Name'].values
    }
    
    print(f"  Index created for {len(embeddings)} samples")
    
    return nn_model, similarity_data

def save_models(embedding_model, models_dict, similarity_model, similarity_data, mlb, 
                normalization_mappings, two_stage_models):
    """Save all trained models and normalization mappings."""
    print("\nSaving models...")
    
    # Save embedding model
    embedding_path = os.path.join(MODELS_DIR, 'embedding_model')
    embedding_model.save(embedding_path)
    print(f"  Saved embedding model to: {embedding_path}")
    
    # Save two-stage models for Dosage Form and Manufacturer
    for model_name, two_stage_model in two_stage_models.items():
        model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(two_stage_model, f)
        print(f"  Saved {model_name} two-stage model to: {model_path}")
    
    # Save single-stage models for Disposal Category and Method of Disposal
    for model_name in ['disposal_category', 'method_of_disposal']:
        if model_name in models_dict:
            model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(models_dict[model_name], f)
            print(f"  Saved {model_name} model to: {model_path}")
            # If an encoder/labeler exists for this model, save it too
            encoder_key = f"{model_name}_le"
            if encoder_key in models_dict and models_dict[encoder_key] is not None:
                enc_path = os.path.join(MODELS_DIR, f'{model_name}_label_encoder.pkl')
                with open(enc_path, 'wb') as ef:
                    pickle.dump(models_dict[encoder_key], ef)
                print(f"  Saved {model_name} label encoder to: {enc_path}")
    
    # Save multi-label binarizer
    if mlb is not None:
        mlb_path = os.path.join(MODELS_DIR, 'multilabel_binarizer.pkl')
        with open(mlb_path, 'wb') as f:
            pickle.dump(mlb, f)
        print(f"  Saved multilabel binarizer to: {mlb_path}")
    
    # Save similarity model and data
    similarity_model_path = os.path.join(MODELS_DIR, 'similarity_model.pkl')
    with open(similarity_model_path, 'wb') as f:
        pickle.dump(similarity_model, f)
    print(f"  Saved similarity model to: {similarity_model_path}")
    
    similarity_data_path = os.path.join(MODELS_DIR, 'similarity_data.pkl')
    with open(similarity_data_path, 'wb') as f:
        pickle.dump(similarity_data, f)
    print(f"  Saved similarity data to: {similarity_data_path}")
    
    # Save normalization mappings
    normalization_path = os.path.join(MODELS_DIR, 'normalization_mappings.pkl')
    with open(normalization_path, 'wb') as f:
        pickle.dump(normalization_mappings, f)
    print(f"  Saved normalization mappings to: {normalization_path}")
    
    print("\n✓ All models saved successfully!")

def main():
    """Main training function."""
    print("=" * 60)
    print("Pharmaceutical Disposal Prediction Model Training (IMPROVED)")
    print("=" * 60)
    
    # Create models directory
    create_models_directory()
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Initialize embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Create embeddings
    embeddings = create_embeddings(df, embedding_model)
    
    # Extract engineered features
    print("\nExtracting engineered features...")
    generic_names = df['Generic Name'].tolist()
    engineered_features = extract_features(generic_names)
    print(f"Created engineered features of shape: {engineered_features.shape}")
    
    # Create combined features (embeddings + engineered features)
    X_combined = create_combined_features(embeddings, engineered_features)
    
    # Prepare outputs with normalization and rare class merging
    X_embeddings = embeddings  # For single-stage models
    X_features = X_combined  # For models with feature engineering
    normalization_mappings = {}
    
    # 1. Dosage Form - standardize and merge rare classes
    print("\n" + "=" * 60)
    print("Preprocessing Dosage Form...")
    y_dosage = df['Dosage Form Standardized'].values
    y_dosage_merged, dosage_mapping = merge_rare_classes(
        pd.Series(y_dosage), 
        min_frequency=MIN_CLASS_FREQUENCY,
        other_label='other'
    )
    y_dosage = y_dosage_merged.values
    normalization_mappings['dosage_form'] = {
        'rare_class_mapping': dosage_mapping,
        'original_to_standardized': dict(zip(df['Dosage Form'], df['Dosage Form Standardized']))
    }
    print(f"  Final unique classes: {len(np.unique(y_dosage))}")
    
    # 2. Manufacturer - normalize and merge rare classes
    print("\n" + "=" * 60)
    print("Preprocessing Manufacturer...")
    y_manufacturer = df['Manufacturer Normalized'].values
    y_manufacturer_merged, mfg_mapping = merge_rare_classes(
        pd.Series(y_manufacturer),
        min_frequency=MIN_CLASS_FREQUENCY,
        other_label='other'
    )
    y_manufacturer = y_manufacturer_merged.values
    normalization_mappings['manufacturer'] = {
        'rare_class_mapping': mfg_mapping,
        'original_to_normalized': dict(zip(df["Manufacturer's Name"], df['Manufacturer Normalized']))
    }
    print(f"  Final unique classes: {len(np.unique(y_manufacturer))}")
    
    # 3. Disposal Category - just normalize (already has few classes)
    print("\n" + "=" * 60)
    print("Preprocessing Disposal Category...")
    y_category = df['Disposal Category Normalized'].values
    normalization_mappings['disposal_category'] = {
        'original_to_normalized': dict(zip(df['Disposal Category'], df['Disposal Category Normalized']))
    }
    print(f"  Final unique classes: {len(np.unique(y_category))}")
    
    # Train models - Two-stage approach only
    print("\n" + "=" * 60)
    print("TRAINING TWO-STAGE MODELS (with feature engineering)")
    print("=" * 60)
    models = {}
    two_stage_models = {}
    
    # Create broad categories
    dosage_broad, manufacturer_broad = create_broad_categories(y_dosage, y_manufacturer)
    
    # 1. Dosage Form two-stage
    two_stage_models['dosage_form'] = train_two_stage_model(
        X_features, dosage_broad, y_dosage, 'Dosage Form',
        n_estimators_stage1=100, n_estimators_stage2=200, use_xgboost=USE_XGBOOST
    )
    
    # 2. Manufacturer two-stage
    two_stage_models['manufacturer'] = train_two_stage_model(
        X_features, manufacturer_broad, y_manufacturer, 'Manufacturer',
        n_estimators_stage1=100, n_estimators_stage2=200, use_xgboost=USE_XGBOOST
    )
    
    # Store broad category mappings for prediction
    normalization_mappings['dosage_form_broad'] = dosage_broad
    normalization_mappings['manufacturer_broad'] = manufacturer_broad
    
    # 3. Disposal Category (single-stage, as it has few classes)
    disp_result = train_categorical_model(
        X_features, y_category, 'Disposal Category', n_estimators=100, use_xgboost=USE_XGBOOST
    )
    # train_categorical_model returns (model, encoder) when encoders used
    if isinstance(disp_result, tuple):
        models['disposal_category'], models['disposal_category_le'] = disp_result
    else:
        models['disposal_category'] = disp_result
    
    # 4. Method of Disposal (Multi-label) - use combined features
    y_multilabel, mlb = prepare_multilabel_data(df)
    models['method_of_disposal'], mlb = train_multilabel_model(
        X_features, y_multilabel, mlb, use_xgboost=USE_XGBOOST
    )
    
    # 5. Create similarity index for text retrieval
    similarity_model, similarity_data = create_similarity_index(df, embeddings)
    
    # Save all models (both single-stage and two-stage)
    save_models(embedding_model, models, similarity_model, similarity_data, mlb, 
                normalization_mappings, two_stage_models=two_stage_models)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nModels saved in: {MODELS_DIR}/")
    print(f"Model type: {'XGBoost' if USE_XGBOOST else 'RandomForest'}")
    print(f"Minimum class frequency: {MIN_CLASS_FREQUENCY}")
    print("\nApproach: Two-stage models (broad category → specific class)")
    print("\nNext steps:")
    print("  1. Run test.py to test the models")
    print("  2. Use the models in your application")

if __name__ == '__main__':
    main()
