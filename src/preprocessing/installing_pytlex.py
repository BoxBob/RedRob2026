import os
import urllib.request
import zipfile
import sys

# Setup stdout for Windows console UTF-8 support
sys.stdout.reconfigure(encoding='utf-8')

# Resolve absolute path to workspace root
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_pytlex():
    url = "https://cognac.cs.fiu.edu/wp-content/uploads/sites/11/2023/11/edu.fiu_.pytlex.zip"
    zip_path = os.path.join(workspace_root, "fiu.edu.pytlex.zip")
    extract_dir = os.path.join(workspace_root, "fiu_pytlex")

    if not os.path.exists(extract_dir):
        print(f"Downloading pyTLEX from {url}...")
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        try:
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Download failed with error: {e}")
            return
        print("Extracting pyTLEX...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        os.remove(zip_path)
        print("pyTLEX setup complete.")
    else:
        print("pyTLEX is already installed locally. Skipping download.")

if __name__ == "__main__":
    setup_pytlex()

# Inject the package root into python path for downstream tasks
project_root = os.path.join(workspace_root, 'fiu_pytlex', 'edu.fiu.pytlex')
sys.path.insert(0, project_root)
