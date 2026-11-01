"""Build the paper, classroom slides and supplementary derivations."""
import subprocess
from pathlib import Path
if __name__ == '__main__':
    subprocess.run(['bash', str(Path(__file__).resolve().parents[2] / 'docs/build_docs.sh')], check=True)
