"""
Script to download IMDB dataset.

Downloads and extracts the IMDB movie review dataset from Stanford AI Lab.
"""

import logging
import os
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_DIR = Path("data")


def download_file(url: str, dest_path: Path) -> None:
    """
    Download file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(dest_path, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_archive(archive_path: Path, extract_path: Path) -> None:
    """
    Extract tar.gz archive.
    
    Args:
        archive_path: Path to archive file
        extract_path: Directory to extract to
    """
    logger.info(f"Extracting {archive_path}...")
    
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_path)
    
    logger.info("Extraction complete")


def reorganize_data(extracted_path: Path, final_path: Path) -> None:
    """
    Reorganize extracted data to expected format.
    
    Args:
        extracted_path: Path to extracted aclImdb directory
        final_path: Final data directory path
    """
    import shutil
    
    train_path = final_path / "train"
    test_path = final_path / "test"
    
    if (extracted_path / "train").exists():
        if train_path.exists():
            shutil.rmtree(train_path)
        shutil.copytree(extracted_path / "train" / "pos", train_path / "pos")
        shutil.copytree(extracted_path / "train" / "neg", train_path / "neg")
        logger.info(f"Train data moved to {train_path}")
    
    if (extracted_path / "test").exists():
        if test_path.exists():
            shutil.rmtree(test_path)
        shutil.copytree(extracted_path / "test" / "pos", test_path / "pos")
        shutil.copytree(extracted_path / "test" / "neg", test_path / "neg")
        logger.info(f"Test data moved to {test_path}")


def main() -> None:
    """Main function to download and prepare IMDB dataset."""
    logger.info("Starting IMDB dataset download...")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if (DATA_DIR / "train" / "pos").exists() and (DATA_DIR / "test" / "pos").exists():
        logger.info("Dataset already exists. Skipping download.")
        
        train_pos = len(list((DATA_DIR / "train" / "pos").glob("*.txt")))
        train_neg = len(list((DATA_DIR / "train" / "neg").glob("*.txt")))
        test_pos = len(list((DATA_DIR / "test" / "pos").glob("*.txt")))
        test_neg = len(list((DATA_DIR / "test" / "neg").glob("*.txt")))
        
        logger.info(f"Train: {train_pos} positive, {train_neg} negative")
        logger.info(f"Test: {test_pos} positive, {test_neg} negative")
        return
    
    archive_path = DATA_DIR / "aclImdb_v1.tar.gz"
    
    if not archive_path.exists():
        logger.info(f"Downloading from {IMDB_URL}...")
        download_file(IMDB_URL, archive_path)
    else:
        logger.info("Archive already downloaded")
    
    extract_archive(archive_path, DATA_DIR)
    
    extracted_path = DATA_DIR / "aclImdb"
    reorganize_data(extracted_path, DATA_DIR)
    
    logger.info("Cleaning up temporary files...")
    import shutil
    if extracted_path.exists():
        shutil.rmtree(extracted_path)
    if archive_path.exists():
        archive_path.unlink()
    
    logger.info("Dataset preparation complete!")
    
    train_pos = len(list((DATA_DIR / "train" / "pos").glob("*.txt")))
    train_neg = len(list((DATA_DIR / "train" / "neg").glob("*.txt")))
    test_pos = len(list((DATA_DIR / "test" / "pos").glob("*.txt")))
    test_neg = len(list((DATA_DIR / "test" / "neg").glob("*.txt")))
    
    logger.info(f"Train: {train_pos} positive, {train_neg} negative")
    logger.info(f"Test: {test_pos} positive, {test_neg} negative")


if __name__ == "__main__":
    main()
