"""
Quick fix script to downgrade NumPy to compatible version.
Run this to fix NumPy 2.0 compatibility issues.
"""

import subprocess
import sys

def fix_numpy():
    """Downgrade NumPy to compatible version."""
    print("Fixing NumPy compatibility issue...")
    print("Downgrading NumPy to < 2.0...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0", "--upgrade"])
        print("\n✓ NumPy downgraded successfully!")
        print("\nYou can now run:")
        print("  python predict.py \"Paracetamol\"")
        print("  python predict.py \"path/to/image.jpg\"")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease manually run:")
        print("  pip install 'numpy<2.0' --upgrade")

if __name__ == '__main__':
    fix_numpy()



