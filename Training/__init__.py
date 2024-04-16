import sys
import os

# Ensuring the package's parent directory is in sys.path
package_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(package_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
