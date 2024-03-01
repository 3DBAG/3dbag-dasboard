import sys
from pathlib import Path
import pandas as pd


def main():
    data_dir = Path(sys.argv[1]).resolve()

    df = pd.read_csv(data_dir / "reconstructed_features.csv", low_memory=False)
    df.to_parquet(data_dir / "reconstructed_features.parquet", engine="pyarrow")

    df = pd.read_csv(data_dir / "validate_compressed_files.csv", low_memory=False)
    df.to_parquet(data_dir / "validate_compressed_files.parquet", engine="pyarrow")


if __name__ == "__main__":
    main()
