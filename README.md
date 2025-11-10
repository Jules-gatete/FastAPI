# Pharmaceutical Disposal Prediction Model

Predict pharmaceutical disposal guidance from a medicine **Generic Name** using embeddings, classical classifiers and nearest-neighbour retrieval.

---

## Quick summary

Inputs:

* Text or image containing a Generic Name (image → OCR).

Outputs:

* **Dosage Form** (top-3 with confidences)
* **Manufacturer** (top-3 with confidences)
* **Disposal Category** (single label + confidence)
* **Method of Disposal** (multi-label + confidences)
* **Handling Method** (retrieved text via similarity search)
* **Disposal Remarks** (retrieved text via similarity search)

Main components:

* Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
* Classifiers: Random Forests (single & multi-output)
* Multi-label: `MultiOutputClassifier` (Random Forest)
* Retrieval: nearest-neighbour search over pre-indexed handling/disposal texts
* OCR (for images): EasyOCR

---

## Features & behaviour

* Accepts text or image input; automatic detection.
* OCR fallback: extracts candidate Generic Name(s) and predicts from them.
* Top-k predictions for Dosage Form & Manufacturer.
* Multi-label predictions for Methods of Disposal.
* Human-readable analysis output or machine-friendly JSON.
* Similarity retrieval for detailed textual guidance (handling method, remarks).

---

## Requirements

* Python 3.7+
* pandas >= 1.5.0
* numpy >= 1.23.0
* scikit-learn >= 1.2.0
* sentence-transformers >= 2.2.0
* easyocr >= 1.7.0
* opencv-python >= 4.8.0
* pillow >= 10.0.0
* joblib (recommended)

Install:

```bash
pip install -r requirements.txt
```

> Note: EasyOCR downloads language model files on first run (~500MB).

---

## File layout (recommended)

```
.
├── datasets/
│   └── disposal_dataset.csv
├── models/
│   ├── embedding_model/                # sentence-transformers folder
│   ├── dosage_form_model.pkl
│   ├── manufacturer_model.pkl
│   ├── disposal_category_model.pkl
│   ├── method_of_disposal_model.pkl
│   ├── multilabel_binarizer.pkl
│   ├── similarity_model.pkl
│   └── similarity_data.pkl
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   └── ocr_utils.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <repo>
cd repo
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Quick usage

### Train everything

```bash
python src/train.py
```

Creates embeddings, trains classifiers, builds retrieval index, and writes models to `models/`.

### Interactive test mode

```bash
python src/predict.py
# then type at prompt:
# Generic Name: Paracetamol
# Generic Name: quit
```

### Batch text predictions

```bash
python src/predict.py "Paracetamol" "Glimepiride, Metformin HCl" "Atorvastatin"
```

### Image prediction

```bash
python src/predict.py "path/to/medicine_cover.jpg"
```

### Output formats

* Default: full textual analysis
* Short: `--format summary`
* JSON: `--format json` or `--json`

---

## Example output

```
================================================================================
PREDICTIONS FOR: Paracetamol
================================================================================

📋 DOSAGE FORM (Top 3):
  1. Tablets                                            (Confidence: 85.23%)
  2. Suspension                                          (Confidence: 12.45%)
  3. Syrup                                               (Confidence: 2.32%)

🏭 MANUFACTURER (Top 3):
  1. MSN LABORATORIES PRIVATE LIMITED                    (Confidence: 45.67%)
  2. BETA HEALTHCARE INTERNATIONAL LTD                   (Confidence: 32.12%)
  3. AJANTA PHARMA LIMITED                               (Confidence: 22.21%)

🗑️  DISPOSAL CATEGORY:
  Solids, Semisolids, Powders (Except Biological Waste)  (Confidence: 95.43%)

♻️  METHOD OF DISPOSAL:
  • Landfill                                             (Confidence: 98.12%)
  • Waste encapsulation                                  (Confidence: 97.45%)
  • Waste inertization                                   (Confidence: 96.78%)

📝 HANDLING METHOD:
  To be removed from outer packaging but remain in inner packaging...

⚠️  DISPOSAL REMARKS:
  No more than 1% of daily municipal waste should be disposed of daily...

🔍 Retrieved from similar Generic Name: "Paracetamol 500mg"
   Similarity distance: 0.0234
```

---

## Implementation notes & best practices

### Embeddings

* Use `sentence-transformers/all-MiniLM-L6-v2` for a fast, compact embedding.
* Persist embeddings to disk (e.g., `.npy`) to avoid re-computing.

### Classification

* Dosage Form and Manufacturer are high-cardinality classification problems:

  * Consider label frequency thresholding (group rare labels into `OTHER`) to improve robustness.
  * Use two-stage classification if you want hierarchical refinement.
* Disposal Category: 7 classes — can typically get high accuracy with RF.
* Multi-label Methods of Disposal:

  * Use `MultiOutputClassifier(RandomForestClassifier())` or `OneVsRestClassifier` with `LogisticRegression`.
  * Save MultiLabelBinarizer to convert between labels & vectors.

### Retrieval (Handling & Remarks)

* Create a vector store of textual handling/remark entries, compute embeddings, and use `NearestNeighbors` (cosine) or `faiss` for speed at scale.
* Return the nearest match and distance; optionally return top-3 matches with aggregated metadata.

### OCR

* Use EasyOCR for image → text extraction.
* Preprocess images: grayscale, denoise, deskew, and increase contrast prior to OCR for better results.

### Evaluation & Metrics

* Dosage Form: top-1/top-3 accuracy, confusion matrix, per-class recall/precision.
* Manufacturer: top-1/top-3 accuracy, per-class metrics; be cautious — manufacturer names are noisy.
* Disposal Category: accuracy, F1-score.
* Multi-label: micro/macro precision, recall, F1; exact match ratio.
* Retrieval: mean reciprocal rank (MRR), recall@k, average distance.

### Data hygiene

* Normalize Generic Name (lowercase, normalize whitespace, remove punctuation except commas).
* Tokenize multi-generic inputs (e.g., “Glimepiride, Metformin HCl”) and predict per-item or handle multi-input logic.
* De-duplicate training records and handle multiple outputs per same Generic Name.

---

## Model storage & format

* Use `joblib.dump` or `pickle` for sklearn models.
* Save embedding model as sentence-transformers folder under `models/embedding_model/`.
* Save:

  * `dosage_form_model.pkl`
  * `manufacturer_model.pkl`
  * `disposal_category_model.pkl`
  * `method_of_disposal_model.pkl`
  * `multilabel_binarizer.pkl`
  * `similarity_model.pkl` (NearestNeighbors)
  * `similarity_data.pkl` (list of text entries + metadata)
* Keep a `models/manifest.json` describing versions/dates/hyperparameters.

---

## Practical tips & caveats

* **High-cardinality labels** (manufacturers, dosage forms): expect moderate accuracy; consider hierarchical models or candidate generation + rerank to improve top-k.
* **Bias / data drift**: manufacturer lists change frequently — plan to retrain periodically.
* **Ambiguous Generic Names**: many generics map to multiple dosage forms and manufacturers — return top-k & similarity hints, and surface the original matched training example.
* **Multi-generic input**: split and predict per-generic to avoid mixed predictions.
* **Logging**: log inputs, model confidence and nearest-neighbour match for manual review & continuous improvement.
* **Safety**: do not provide medical advice — this is operational disposal guidance only.

---

## Evaluation checklist before production

* [ ] Hold-out test set with realistic distribution
* [ ] Top-3 accuracy metrics logged for Dosage & Manufacturer
* [ ] Class-wise precision/recall for Disposal Category
* [ ] Micro/macro F1 for multi-label Methods of Disposal
* [ ] Retrieval quality: MRR & recall@k
* [ ] Human review of retrieved Handling Methods / Remarks for correctness
* [ ] End-to-end integration tests for OCR → predict pipeline

---

## Packaging & deployment suggestions

* Wrap predict logic in an API (FastAPI) returning JSON for integrations.
* Expose both textual and JSON output modes.
* Containerize with Docker and include pre-warm steps for embedding model & nearest neighbour index.
* Add a small web UI for human-in-the-loop correction (accept/reject suggestions) to collect more labelled data.
* Consider moving retrieval to Faiss or ElasticSearch for larger datasets.

---

## Troubleshooting

* **Embedding model not found**: ensure `models/embedding_model` exists or re-run training.
* **OCR returns noise**: add image preprocessing steps (denoise, deskew, threshold).
* **Low accuracy**: increase training data, group low-frequency labels, or use candidate generation + neural reranker.
* **Slow predictions**: reduce `n_estimators`, use a smaller RandomForest, batch embeddings, or convert to a lightweight model (e.g., LightGBM).
