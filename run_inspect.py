from predict import predict_from_input
from collections.abc import Mapping, Sequence
import numpy as np

def scan(obj, path=''):
    if isinstance(obj, np.generic):
        print(f"{path}: numpy scalar -> {type(obj)} -> value={obj}")
    elif isinstance(obj, np.ndarray):
        print(f"{path}: numpy array -> {type(obj)}, shape={obj.shape}")
    elif isinstance(obj, Mapping):
        for k,v in obj.items():
            scan(v, f"{path}/{k}")
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for i,v in enumerate(obj):
            scan(v, f"{path}[{i}]")
    else:
        try:
            mod = obj.__class__.__module__
            if isinstance(mod, str) and 'numpy' in mod:
                print(f"{path}: {obj} -> {obj.__class__}")
        except Exception:
            pass

if __name__ == '__main__':
    res = predict_from_input('paracetamol')
    print('Result keys:', list(res.keys()))
    import json
    try:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as e:
        print('JSON dump error:', e)
    print('\nScanning for numpy types...')
    scan(res)
