"""Prepare files for HuggingFace Spaces deployment."""

import shutil
from pathlib import Path

def prepare_deployment():
    """Copy necessary files to huggingface/ directory."""
    hf_dir = Path("huggingface")
    hf_dir.mkdir(exist_ok=True)
    
    # Copy src directory
    src_dest = hf_dir / "src"
    if src_dest.exists():
        shutil.rmtree(src_dest)
    shutil.copytree("src", src_dest)
    
    # Copy data/processed (parquet files + lexical corpus)
    data_dest = hf_dir / "data" / "processed"
    data_dest.mkdir(parents=True, exist_ok=True)
    
    processed_dir = Path("data/processed")
    for file in processed_dir.glob("*"):
        shutil.copy(file, data_dest / file.name)
        print(f"Copied {file.name}")
    
    print(f"\nDeployment files prepared in {hf_dir}/")
    print("\nNext steps:")
    print("1. Create a new Space on huggingface.co/spaces")
    print("2. Clone the Space repo")
    print("3. Copy contents of huggingface/ to the Space")
    print("4. Add secrets in Space settings:")
    print("   - XAI_API_KEY")
    print("   - QDRANT_URL")
    print("   - QDRANT_API_KEY")
    print("5. Push to deploy")

if __name__ == "__main__":
    prepare_deployment()

