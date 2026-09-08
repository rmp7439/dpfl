import os
import glob
import re

def fix_file(filepath, is_script_or_test=True):
    with open(filepath, 'r') as f:
        content = f.read()

    # Add sys.path modification for scripts and tests
    if is_script_or_test and "sys.path.insert(0, os.path.abspath" not in content:
        imports_end = content.find('\n\n')
        if imports_end == -1: imports_end = len(content)
        
        # Inject sys.path modification
        injection = "\nimport sys\nimport os\nproject_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))\nif project_root not in sys.path: sys.path.insert(0, project_root)\nsrc_path = os.path.join(project_root, 'src')\nif src_path not in sys.path: sys.path.insert(0, src_path)\n"

        
        # We need to ensure we don't duplicate imports if they exist
        content = injection + content

    # Replace absolute imports
    content = re.sub(r'^import config', 'from dpfl import config', content, flags=re.MULTILINE)
    content = re.sub(r'^from config import', 'from dpfl.config import', content, flags=re.MULTILINE)
    content = re.sub(r'^from model import', 'from dpfl.model import', content, flags=re.MULTILINE)
    content = re.sub(r'^from data_split import', 'from dpfl.data import', content, flags=re.MULTILINE)
    
    # In scripts/run_federated.py, it imports from train_baseline
    content = re.sub(r'^from train_baseline import', 'from scripts.train_baseline import', content, flags=re.MULTILINE)
    
    # In tests, it might import from federated
    content = re.sub(r'^from federated import', 'from scripts.run_federated import', content, flags=re.MULTILINE)

    with open(filepath, 'w') as f:
        f.write(content)

# Fix scripts
for f in glob.glob("scripts/*.py"):
    fix_file(f, is_script_or_test=True)

# Fix tests
for f in glob.glob("tests/*.py"):
    fix_file(f, is_script_or_test=True)

# Fix src
for f in glob.glob("src/dpfl/*.py"):
    fix_file(f, is_script_or_test=False)
