from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field

class PlasmidSettings(BaseSettings):
    """Configuration for Plasmid Pretraining Data Pipeline."""
    
    # Base Paths
    base_dir: Path = Field(default=Path("./plasmid_data"), description="Root directory for data processing")
    input_dir: Path = Field(default=Path("./tests/data"), description="Input directory containing tars or raw files")
    
    # Explicit Step Outputs (acting as cache keys/locations)
    # If not provided, they default to subdirectories of base_dir
    raw_dir: Optional[Path] = None
    parsed_dir: Optional[Path] = None
    annotated_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    final_dir: Optional[Path] = None

    # Processing Params
    batch_size: int = Field(default=1000, description="Number of files per processing batch")
    sequence_ratio: float = Field(default=0.1, description="Ratio of sequence-based inserts vs name-based")
    
    # Split Ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    random_seed: int = 42
    force_reprocess: bool = Field(default=False, description="Ignore caches and re-run all steps")

    def model_post_init(self, __context):
        """Set defaults for subdirectories if not explicitly provided."""
        if self.raw_dir is None:
            self.raw_dir = self.base_dir / "0_raw"
        if self.parsed_dir is None:
            self.parsed_dir = self.base_dir / "1_parsed"
        if self.annotated_dir is None:
            self.annotated_dir = self.base_dir / "2_annotated"
        if self.processed_dir is None:
            self.processed_dir = self.base_dir / "3_processed"
        if self.final_dir is None:
            self.final_dir = self.base_dir / "4_final"
            
        # Ensure directories exist
        for d in [self.raw_dir, self.parsed_dir, self.annotated_dir, self.processed_dir, self.final_dir]:
            d.mkdir(parents=True, exist_ok=True)

settings = PlasmidSettings()
