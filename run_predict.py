import traceback, json
try:
    import test
    models = test.load_models()
    preds = test.predict_all(models, 'paracetamol')
    print(json.dumps(preds, indent=2, default=str))
except Exception as e:
    traceback.print_exc()
