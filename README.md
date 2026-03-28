# MoE Project Structure

src/
    data/
        __init__.py
        acquisition.py      # Data loading and acquisition
        preprocessing.py    # Preprocessing, missing modality handling
        features.py         # Feature extraction for audio, physiological, context
    utils/
        __init__.py
        alignment.py        # Timestamp alignment utilities
        segmentation.py     # Sliding window segmentation
    notebooks/
        # For exploratory analysis
    tests/
        # Unit tests

README.md

# Stage 1: Data Acquisition & Feature Engineering

- src/data/acquisition.py: Load DEAP dataset, validate modalities
- src/notebooks/preprocessing.ipynb: Handle missing/corrupted modalities, Align modalities to unified temporal, Implement sliding window segmentationresolution
- src/noteboooks/features.ipynb: Extract features for each modality

# Next Steps
- Create starter files for each module
- Implement K-EmoCon loader and basic feature extraction
