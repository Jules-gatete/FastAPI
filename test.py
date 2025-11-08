"""
Test the trained ML models for pharmaceutical disposal prediction (Two-Stage Approach).
Takes Generic Name as input and predicts all outputs.
Uses two-stage approach for Dosage Form and Manufacturer.
Handles text normalization automatically.
"""

import pickle
import os
import csv
from pathlib import Path
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Configuration
MODELS_DIR = 'models'
BASE_DIR = Path(__file__).resolve().parent
DISPOSAL_METHOD_FILE = BASE_DIR / 'disposal_method.csv'


def _load_disposal_category_lookup():
    """Load disposal category metadata from disposal_method.csv.

    Returns a dict keyed by the code (as string) containing human readable
    category details plus recommended handling / disposal guidance.
    """
    if not DISPOSAL_METHOD_FILE.exists():
        return {}

    lookup = {}
    try:
        with DISPOSAL_METHOD_FILE.open(mode='r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = str(row.get('S. No.', '')).strip()
                if not code:
                    continue

                lookup[code] = {
                    'category': (row.get('Category of Drug/Dosage Form/Packaging') or '').strip(),
                    'handling_method': (row.get('Handling Method') or '').strip(),
                    'disposal_method': (row.get('Method of Disposal') or '').strip(),
                    'remarks': (row.get('Remarks') or '').strip(),
                }
    except Exception:
        return {}

    return lookup

def normalize_text(text):
    """Normalize text: lowercase, strip, remove extra spaces."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ''
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

def title_case(text):
    """Convert text to title case for better readability."""
    if not text or text == 'other':
        return text
    return text.title()

def extract_features(generic_names):
    """
    Extract engineered features from Generic Names (same as in train.py).
    Returns a feature matrix with engineered features.
    """
    import numpy as np
    import re
    
    if isinstance(generic_names, str):
        generic_names = [generic_names]
    
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

def load_models():
    """Load all trained models."""
    print("Loading models...")
    
    models = {}
    
    # Load embedding model
    embedding_path = os.path.join(MODELS_DIR, 'embedding_model')
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding model not found at: {embedding_path}")
    models['embedding'] = SentenceTransformer(embedding_path)
    print("  ✓ Loaded embedding model")
    
    # Load classification models
    model_names = ['dosage_form', 'manufacturer', 'disposal_category', 'method_of_disposal']
    for model_name in model_names:
        model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
        print(f"  ✓ Loaded {model_name} model")
    
    # Load multi-label binarizer
    mlb_path = os.path.join(MODELS_DIR, 'multilabel_binarizer.pkl')
    if not os.path.exists(mlb_path):
        raise FileNotFoundError(f"Multi-label binarizer not found at: {mlb_path}")
    with open(mlb_path, 'rb') as f:
        models['mlb'] = pickle.load(f)
    print("  ✓ Loaded multi-label binarizer")
    
    # Load similarity model and data
    similarity_model_path = os.path.join(MODELS_DIR, 'similarity_model.pkl')
    similarity_data_path = os.path.join(MODELS_DIR, 'similarity_data.pkl')
    
    if not os.path.exists(similarity_model_path):
        raise FileNotFoundError(f"Similarity model not found at: {similarity_model_path}")
    if not os.path.exists(similarity_data_path):
        raise FileNotFoundError(f"Similarity data not found at: {similarity_data_path}")
    
    with open(similarity_model_path, 'rb') as f:
        models['similarity'] = pickle.load(f)
    with open(similarity_data_path, 'rb') as f:
        models['similarity_data'] = pickle.load(f)
    print("  ✓ Loaded similarity search models")
    
    # Load normalization mappings (if available)
    normalization_path = os.path.join(MODELS_DIR, 'normalization_mappings.pkl')
    if os.path.exists(normalization_path):
        with open(normalization_path, 'rb') as f:
            models['normalization_mappings'] = pickle.load(f)
        print("  ✓ Loaded normalization mappings")
    else:
        models['normalization_mappings'] = None
        print("  ⚠ Normalization mappings not found (using old model format)")

    # Load disposal category metadata (optional)
    disposal_lookup = _load_disposal_category_lookup()
    if disposal_lookup:
        models['disposal_category_lookup'] = disposal_lookup
        print("  ✓ Loaded disposal category metadata")
    else:
        models['disposal_category_lookup'] = {}
        print("  ⚠ Disposal category metadata not found (predictions will show raw codes)")
    
    # Load two-stage models for Dosage Form and Manufacturer
    two_stage_models = {}
    for model_name in ['dosage_form', 'manufacturer']:
        model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                two_stage_models[model_name] = pickle.load(f)
    if two_stage_models:
        models['two_stage_models'] = two_stage_models
        print(f"  ✓ Loaded {len(two_stage_models)} two-stage models")
    else:
        raise FileNotFoundError("Two-stage models not found. Please run train.py first.")
    
        # Try loading label encoders for single-stage models (optional)
        disp_le_path = os.path.join(MODELS_DIR, 'disposal_category_label_encoder.pkl')
        if os.path.exists(disp_le_path):
            try:
                with open(disp_le_path, 'rb') as f:
                    models['disposal_category_le'] = pickle.load(f)
                print("  ✓ Loaded disposal_category label encoder")
            except Exception:
                models['disposal_category_le'] = None
        else:
            models['disposal_category_le'] = None
    
    print("\nAll models loaded successfully!\n")
    return models


def _safe_predict_proba(model, X):
    """Call predict_proba on model with a safe fallback.

    Some pickled models may raise unexpected AttributeError (e.g. compatibility
    issues that surface as missing internal attributes like 'monotonic_cst').
    This wrapper attempts predict_proba and falls back to predict when needed,
    returning a probability-like array where the predicted class gets 1.0.
    """
    import numpy as _np
    try:
        # Ensure any tree-based nested estimators have compatibility attributes
        try:
            _ensure_tree_compat(model)
        except Exception:
            pass
        return model.predict_proba(X)
    except AttributeError as e:
        # Return a probability array derived from predict()
        try:
            preds = model.predict(X)
        except Exception:
            # As a last resort, return zeros
            return _np.zeros((X.shape[0], 1))

        # If model exposes classes_, construct a one-hot style probability
        if hasattr(model, 'classes_'):
            classes = getattr(model, 'classes_')
            probs = _np.zeros((X.shape[0], len(classes)))
            for i, p in enumerate(preds):
                # find index in classes; if not found, put in last position
                try:
                    idx = int(_np.where(classes == p)[0][0])
                except Exception:
                    idx = -1
                probs[i, idx] = 1.0
            return probs
        else:
            # Unknown class structure: return a single-column probability with 1.0 for predicted
            probs = _np.ones((X.shape[0], 1))
            return probs


def _ensure_tree_compat(model):
    """Ensure tree-based estimators have attributes expected by this sklearn version.

    When unpickling models trained with an older/newer scikit-learn version, some
    attributes (e.g. 'monotonic_cst') may be missing on DecisionTreeClassifier
    instances. This function tries to set those attributes to safe defaults so
    predict()/predict_proba() calls don't raise AttributeError.
    """
    # Handle ensemble wrappers (RandomForest, Bagging, etc.)
    def _patch_obj(obj):
        try:
            # If object is an ensemble with its own estimators_, recurse into them
            if hasattr(obj, 'estimators_') and obj.estimators_ is not None:
                for sub in obj.estimators_:
                    _patch_obj(sub)
            else:
                # If it's a DecisionTree-like object, ensure monotonic_cst exists
                if not hasattr(obj, 'monotonic_cst'):
                    try:
                        setattr(obj, 'monotonic_cst', None)
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        _patch_obj(model)
    except Exception:
        pass
    # For nested dicts of stage2 models we handle where used

def predict_all(models, generic_name, top_k=3):
    """
    Predict all outputs for a given Generic Name using two-stage approach.
    
    Args:
        models: Dictionary of loaded models
        generic_name: Input Generic Name (string)
        top_k: Number of top predictions to return for categorical outputs
    
    Returns:
        Dictionary with all predictions
    """
    # Normalize input Generic Name
    generic_name_normalized = normalize_text(generic_name)
    
    # Create embedding
    embedding = models['embedding'].encode([generic_name_normalized], convert_to_numpy=True)
    embedding = embedding.reshape(1, -1)
    
    # Extract engineered features
    engineered_features = extract_features([generic_name_normalized])
    
    # Combine embeddings with engineered features
    X_features = np.hstack([embedding, engineered_features])
    
    predictions = {}
    
    # 1. Predict Dosage Form (two-stage approach)
    two_stage_model = models['two_stage_models']['dosage_form']
    # Stage 1: Predict broad category
    # Ensure any missing tree attributes are patched before prediction
    try:
        _ensure_tree_compat(two_stage_model['stage1'])
    except Exception:
        pass
    raw_pred = two_stage_model['stage1'].predict(X_features)[0]
    # If a label encoder was saved with the two-stage model, inverse transform
    if 'stage1_le' in two_stage_model and two_stage_model['stage1_le'] is not None:
        try:
            le = two_stage_model['stage1_le']
            import numpy as _np
            if isinstance(raw_pred, (list, tuple, _np.ndarray)):
                pred_val = _np.array(raw_pred)
            else:
                pred_val = _np.array([raw_pred])
            broad_category = le.inverse_transform(pred_val.astype(int))[0]
        except Exception:
            # Fallback: use raw value directly
            broad_category = raw_pred
    else:
        broad_category = raw_pred
    # Stage 2: Predict specific within category
    if broad_category in two_stage_model['stage2']:
        stage2_model = two_stage_model['stage2'][broad_category]
        # Use safe wrapper to handle unexpected AttributeError from some model instances
        dosage_proba_all = _safe_predict_proba(stage2_model, X_features)
        # Ensure we can index the first (and only) sample
        dosage_proba = dosage_proba_all[0]
        # Try to get class labels from the estimator; if absent, check for a saved encoder
        dosage_classes = getattr(stage2_model, 'classes_', None)
        # If classes_ are numeric (because the estimator was trained on encoded labels)
        # try to decode them using saved stage2 label encoder
        try:
            if dosage_classes is not None:
                # detect numeric classes
                if np.issubdtype(dosage_classes.dtype, np.integer):
                    encoders = two_stage_model.get('stage2_le', {})
                    if encoders and broad_category in encoders:
                        le = encoders[broad_category]
                        try:
                            dosage_classes = le.inverse_transform(dosage_classes.astype(int))
                        except Exception:
                            dosage_classes = dosage_classes.astype(str)
                    else:
                        dosage_classes = dosage_classes.astype(str)
            else:
                encoders = two_stage_model.get('stage2_le', {})
                if encoders and broad_category in encoders:
                    dosage_classes = encoders[broad_category].classes_
                else:
                    dosage_classes = np.array(['unknown'])
        except Exception:
            dosage_classes = np.array(['unknown'])
    else:
        # Fallback: return empty predictions if category not found
        dosage_proba = np.array([0.0])
        dosage_classes = np.array(['unknown'])
    
    top_dosage_indices = np.argsort(dosage_proba)[-top_k:][::-1]
    predictions['dosage_form'] = [
        {
            'value': dosage_classes[idx],
            'confidence': float(dosage_proba[idx])
        }
        for idx in top_dosage_indices
    ]
    
    # 2. Predict Manufacturer (two-stage approach)
    two_stage_model = models['two_stage_models']['manufacturer']
    # Stage 1: Predict broad category
    try:
        _ensure_tree_compat(two_stage_model['stage1'])
    except Exception:
        pass
    raw_pred = two_stage_model['stage1'].predict(X_features)[0]
    if 'stage1_le' in two_stage_model and two_stage_model['stage1_le'] is not None:
        try:
            le = two_stage_model['stage1_le']
            import numpy as _np
            if isinstance(raw_pred, (list, tuple, _np.ndarray)):
                pred_val = _np.array(raw_pred)
            else:
                pred_val = _np.array([raw_pred])
            broad_category = le.inverse_transform(pred_val.astype(int))[0]
        except Exception:
            broad_category = raw_pred
    else:
        broad_category = raw_pred
    # Stage 2: Predict specific within category
    if broad_category in two_stage_model['stage2']:
        stage2_model = two_stage_model['stage2'][broad_category]
        mfg_proba_all = _safe_predict_proba(stage2_model, X_features)
        mfg_proba = mfg_proba_all[0]
        mfg_classes = getattr(stage2_model, 'classes_', None)
        try:
            if mfg_classes is not None:
                if np.issubdtype(mfg_classes.dtype, np.integer):
                    encoders = two_stage_model.get('stage2_le', {})
                    if encoders and broad_category in encoders:
                        le = encoders[broad_category]
                        try:
                            mfg_classes = le.inverse_transform(mfg_classes.astype(int))
                        except Exception:
                            mfg_classes = mfg_classes.astype(str)
                    else:
                        mfg_classes = mfg_classes.astype(str)
            else:
                encoders = two_stage_model.get('stage2_le', {})
                if encoders and broad_category in encoders:
                    mfg_classes = encoders[broad_category].classes_
                else:
                    mfg_classes = np.array(['unknown'])
        except Exception:
            mfg_classes = np.array(['unknown'])
    else:
        # Fallback: return empty predictions if category not found
        mfg_proba = np.array([0.0])
        mfg_classes = np.array(['unknown'])
    
    top_mfg_indices = np.argsort(mfg_proba)[-top_k:][::-1]
    predictions['manufacturer'] = [
        {
            'value': mfg_classes[idx],
            'confidence': float(mfg_proba[idx])
        }
        for idx in top_mfg_indices
    ]
    
    # 3. Predict Disposal Category (use combined features)
    category_proba_all = _safe_predict_proba(models['disposal_category'], X_features)
    category_proba = category_proba_all[0]
    category_idx = np.argmax(category_proba)
    category_classes = getattr(models['disposal_category'], 'classes_', None)
    try:
        if category_classes is not None:
            if np.issubdtype(category_classes.dtype, np.integer):
                # decode using saved disposal_category encoder if present
                enc = models.get('disposal_category_le')
                if enc is not None:
                    try:
                        category_classes = enc.inverse_transform(category_classes.astype(int))
                    except Exception:
                        category_classes = category_classes.astype(str)
                else:
                    category_classes = category_classes.astype(str)
        else:
            category_classes = np.array(['unknown'])
    except Exception:
        category_classes = np.array(['unknown'])
    raw_category = category_classes[category_idx]
    category_code = str(raw_category)
    disposal_lookup = models.get('disposal_category_lookup') or {}
    meta = disposal_lookup.get(category_code)
    display_value = meta.get('category') if meta and meta.get('category') else raw_category
    display_value = str(display_value) if display_value is not None else ''
    raw_category_str = str(raw_category)

    disposal_entry = {
        'code': category_code,
        'value': display_value,
        'raw_value': raw_category_str,
        'confidence': float(category_proba[category_idx])
    }

    if meta:
        if meta.get('handling_method'):
            disposal_entry['handling_method'] = meta['handling_method']
        if meta.get('disposal_method'):
            disposal_entry['recommended_disposal'] = meta['disposal_method']
        if meta.get('remarks'):
            disposal_entry['remarks'] = meta['remarks']

    predictions['disposal_category'] = disposal_entry
    
    # 4. Predict Method of Disposal (multi-label, use combined features)
    # Ensure compatibility for tree-based estimators inside the multi-output classifier
    try:
        _ensure_tree_compat(models['method_of_disposal'])
    except Exception:
        pass
    method_pred = models['method_of_disposal'].predict(X_features)[0]
    method_proba_list = _safe_predict_proba(models['method_of_disposal'], X_features)

    # Normalize different possible shapes returned by predict_proba:
    # - MultiOutputClassifier.predict_proba -> list of arrays (one per label)
    # - Single multi-label estimator -> ndarray of shape (n_samples, n_labels)
    method_labels = models['mlb'].classes_
    predicted_methods = []

    # Helper to extract per-label probability for the first sample
    def _get_method_prob(mpl, label_idx):
        # If list-like (MultiOutputClassifier), follow the old logic
        try:
            if isinstance(mpl, list):
                # mpl[i] -> array of shape (n_samples, n_classes)
                prob_array = mpl[label_idx][0]
                return float(prob_array[1]) if len(prob_array) > 1 else 1.0
        except Exception:
            pass

        # If numpy array with shape (n_samples, n_labels) or (n_samples, n_classes)
        try:
            import numpy as _np
            mpl_arr = _np.array(mpl)
            if mpl_arr.ndim == 2 and mpl_arr.shape[0] >= 1:
                # If columns match number of labels, assume columns are per-label probabilities
                if mpl_arr.shape[1] == len(method_labels):
                    return float(mpl_arr[0, label_idx])
                # Else, if columns==2 and list of label-wise binary probs was flattened,
                # we cannot reliably map; fall back to 1.0
        except Exception:
            pass

        # Fallback
        return 1.0

    for i, label in enumerate(method_labels):
        if i < len(method_pred) and method_pred[i] == 1:
            prob = _get_method_prob(method_proba_list, i)
            predicted_methods.append({'value': label, 'confidence': prob})
    predictions['method_of_disposal'] = predicted_methods
    
    # 5. Retrieve Handling Method and Disposal Remarks using similarity search (use embeddings only)
    distances, indices = models['similarity'].kneighbors(embedding, n_neighbors=1)
    nearest_idx = indices[0][0]
    
    predictions['handling_method'] = models['similarity_data']['handling_methods'][nearest_idx]
    predictions['disposal_remarks'] = models['similarity_data']['disposal_remarks'][nearest_idx]
    predictions['similar_generic_name'] = models['similarity_data']['generic_names'][nearest_idx]
    predictions['similarity_distance'] = float(distances[0][0])
    
    # Store original input for display
    predictions['input_generic_name'] = generic_name
    
    return predictions

def display_predictions(generic_name, predictions):
    """Display predictions in a user-friendly format."""
    print("=" * 80)
    print(f"PREDICTIONS FOR: {generic_name}")
    print("=" * 80)
    
    # Dosage Form (convert to title case for readability)
    print("\n📋 DOSAGE FORM (Top 3):")
    for i, pred in enumerate(predictions['dosage_form'], 1):
        value = title_case(pred['value'])
        print(f"  {i}. {value:<50} (Confidence: {pred['confidence']:.2%})")
    
    # Manufacturer (convert to title case for readability)
    print("\n🏭 MANUFACTURER (Top 3):")
    for i, pred in enumerate(predictions['manufacturer'], 1):
        value = title_case(pred['value'])
        print(f"  {i}. {value:<50} (Confidence: {pred['confidence']:.2%})")
    
    # Disposal Category (convert to title case for readability)
    print("\n🗑️  DISPOSAL CATEGORY:")
    cat = predictions['disposal_category']
    value = title_case(cat['value'])
    print(f"  {value:<50} (Confidence: {cat['confidence']:.2%})")
    
    # Method of Disposal (convert to title case for readability)
    print("\n♻️  METHOD OF DISPOSAL:")
    if predictions['method_of_disposal']:
        for pred in predictions['method_of_disposal']:
            value = title_case(pred['value'])
            print(f"  • {value:<50} (Confidence: {pred['confidence']:.2%})")
    else:
        print("  No methods predicted")
    
    # Handling Method
    print("\n📝 HANDLING METHOD:")
    print(f"  {predictions['handling_method']}")
    
    # Disposal Remarks
    print("\n⚠️  DISPOSAL REMARKS:")
    print(f"  {predictions['disposal_remarks']}")
    
    # Similarity info
    similar_name = predictions.get('similar_generic_name', 'N/A')
    print(f"\n🔍 Retrieved from similar Generic Name: \"{similar_name}\"")
    print(f"   Similarity distance: {predictions['similarity_distance']:.4f}")
    
    print("\n" + "=" * 80)

def interactive_mode(models):
    """Run interactive prediction mode."""
    print("\n" + "=" * 80)
    print("INTERACTIVE PREDICTION MODE (Two-Stage Approach)")
    print("=" * 80)
    print("Enter Generic Name to predict (or 'quit' to exit)")
    print()
    
    while True:
        try:
            generic_name = input("Generic Name: ").strip()
            
            if generic_name.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break
            
            if not generic_name:
                print("Please enter a valid Generic Name.\n")
                continue
            
            # Make predictions
            predictions = predict_all(models, generic_name)
            
            # Display results
            display_predictions(generic_name, predictions)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

def batch_test_mode(models, test_names):
    """Test multiple Generic Names at once."""
    print("\n" + "=" * 80)
    print("BATCH TEST MODE (Two-Stage Approach)")
    print("=" * 80)
    
    for generic_name in test_names:
        try:
            predictions = predict_all(models, generic_name)
            display_predictions(generic_name, predictions)
            print()
        except Exception as e:
            print(f"Error predicting for '{generic_name}': {e}\n")

def main():
    """Main test function."""
    import sys
    
    # Load models
    try:
        models = load_models()
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print("\nMake sure you have run train.py first!")
        return
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        # Batch test mode
        test_names = sys.argv[1:]
        batch_test_mode(models, test_names)
    else:
        # Interactive mode
        interactive_mode(models)

if __name__ == '__main__':
    main()

