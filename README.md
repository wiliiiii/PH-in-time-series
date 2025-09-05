# Time Series

This repository contains Python scripts for studying time series data using **persistent homology** and related tools.  
It includes implementations for Takens' embedding, persistence diagrams, and coefficient field dependence analysis.

---

## 📂 Files

- **MainCodes.py**  
  Main pipeline for:
  - Generating time series datasets
  - Find optimal parameters(embedding dimension and time delay)
  - Performing Takens' embedding
  - Applying PCA for visualization
  - Computing persistence diagrams (using Giotto-TDA)

- **Quantities.py**  
  Utility functions to compute:
  - L² and L³ norms of barcodes
  - Maximum persistence
  - Other numerical invariants

- **VerifyCoeffDependence.py**  
  Script to verify the dependence/independence of persistence diagrams  
  on the choice of coefficient field.

---

## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUserName/PH-in-time-series.git
   cd PH-in-time-series


2.Run the main pipeline:
  ```bash
  python MainCodes.py```


3.Compute numerical quantities:
  ```bash
  python Quantities.py```


4.Verify coefficient field dependence:
  ```bash
  python VerifyCoeffDependence.py```

