# MixedLinear: A DLinear and TimeMixer-Based Ensemble Model for Efficient Electricity Demand Forecasting
### An innovative ensemble model combining DLinear and TimeMixer components for high-accuracy, computationally efficient electricity demand forecasting.

---

## Overview
This project introduces MixedLinear, a novel ensemble approach that leverages the strengths of both DLinear and TimeMixer architectures for enhanced electricity demand forecasting performance.

---

## Related Works

- **DLinear**: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)

- **TimeMixer**: "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting" (ICLR 2024)

---

## Model Description
MixedLinear combines the computational efficiency of DLinear with TimeMixer's ability to handle complex patterns. The model demonstrates superior performance in electricity demand forecasting while maintaining resource efficiency.

<div align="center">
  <img src="https://github.com/user-attachments/assets/afb0a4e5-fba2-4004-b80c-f8a3c05da541" width="600">
</div>

---

## Repository Structure

```
Mixed_Linear/
│
├── models/                      # Model definitions
│   ├── Mixed_Linear.py         # Main Mixed Linear model
│   ├── DLinear.py              # DLinear model
│   ├── NLinear.py              # NLinear model
│   ├── Linear.py               # Linear model
│   ├── TimeMixer.py            # TimeMixer model
│   └── Transformer.py          # Transformer model
│
├── layers/                      # Model layers
│   ├── Autoformer_EncDec.py
│   ├── Embed.py
│   ├── SelfAttention_Family.py
│   ├── StandardNorm.py
│   └── Transformer_EncDec.py
│
├── exp/                         # Experiment framework
│   ├── exp_main.py             # Main experiment class
│   └── exp_basic.py            # Base experiment class
│
├── data_provider/               # Data loaders
│   ├── data_factory.py
│   └── data_loader.py
│
├── utils/                       # Model training/evaluation utilities
│   ├── tools.py                # Learning rate adjustment, EarlyStopping
│   ├── metrics.py              # Evaluation metrics (MAE, MSE, RMSE, etc.)
│   ├── timefeatures.py         # Time feature extraction
│   └── masking.py              # Masking utilities
│
├── utils2/                      # Project-specific utilities
│   ├── Data_run.py             # Unified model execution script
│   └── Date_Processor.py       # Date filtering, preprocessing, PCA
│
├── Notebooks/
│   ├── 01_Data_Collection.ipynb          # Data collection
│   ├── 02_Data_Preprocessing.ipynb       # Data preprocessing
│   ├── 03_Model_Training.ipynb           # Model training (main)
│   ├── 04_Additional_Data_Processing.ipynb # Additional processing
│   ├── 05_Result_Visualization.ipynb     # Result visualization
│   ├── EDA.ipynb                         # Exploratory data analysis
│   └── log_reader.ipynb                  # Log file reader
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT License
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jinsoo96/MixedLinear_TimeSeriesForecasting.git
cd MixedLinear_TimeSeriesForecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Model

**Using Jupyter Notebook:**
```bash
# Open notebooks in sequence:
# 01_Data_Collection.ipynb          # Step 1: Data collection
# 02_Data_Preprocessing.ipynb       # Step 2: Data preprocessing
# 03_Model_Training.ipynb           # Step 3: Model training (main)
# 04_Additional_Data_Processing.ipynb # Step 4: Additional processing
# 05_Result_Visualization.ipynb     # Step 5: Result visualization
```

**Using Python Script:**
```python
from utils2.Data_run import arg_set, model_run

# Configure experiment
args = arg_set(
    folder_path='./Data_Final/1_year',
    data='your_data.csv',
    model_name='Mixed_Linear_experiment',
    pred_len=72,    # Prediction horizon
    label_len=72,   # Label length
    num_workers=10  # Data loader workers
)

# Run model
model_run(args)
```

---

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Pandas
- scikit-learn
- matplotlib

See `requirements.txt` for full dependencies.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Author**: Jin Soo Kim
- **GitHub**: [@jinsoo96](https://github.com/jinsoo96)

---

**Copyright © 2025 Jin Soo Kim All rights reserved.**
