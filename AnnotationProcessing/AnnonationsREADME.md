#  Seizure Chunk Extraction from Expert Annotations

This repository provides tools to extract seizure segments from binary time-series annotations provided by multiple experts (e.g., Experts A, B, C, and CSA). These annotations typically indicate seizure presence (`1`) or absence (`0`) per second for multiple infants or patients.

---

## Features

- Extracts seizure intervals per expert, per infant.
- Converts time (in seconds) to sample indices using a configurable sampling rate (default: 256 Hz).
- Supports batch processing of multiple experts.
- Outputs a clean, concatenated DataFrame or CSV file.

---

## Function Overview

### `find_seizures(expert, data)`

**Purpose**  
Identifies and extracts continuous seizure segments from a DataFrame of binary annotations.

**Parameters**

| Name   | Type           | Description                             |
|--------|----------------|-----------------------------------------|
| `expert` | `str`        | Name of the annotating expert (e.g., "A", "B", "CSA") |
| `data`   | `pd.DataFrame` | Columns represent patients; rows contain binary annotations per second |

**Returns**  
`pd.DataFrame` with the following columns:
- `expert` *(optional if CSA only)*
- `infant`
- `seizure_duration`
- `from_sec`
- `to_sec`
- `from_sample`
- `to_sample`

**Assumptions**
- The sampling rate is 256 Hz.
- Each row in `data` represents one second of annotation.

---

## Example Usage

```python
import pandas as pd
from seizure_extraction import find_seizures  # Assume you saved the function in seizure_extraction.py

# expert_data_list contains four DataFrames: data for A, B, C, and CSA
expert_data_list = [data_A, data_B, data_C, data_CSA]

# Extract seizures per expert
seizures_df_expert_A = find_seizures("A", expert_data_list[0])
seizures_df_expert_B = find_seizures("B", expert_data_list[1])
seizures_df_expert_C = find_seizures("C", expert_data_list[2])
seizures_df_expert_CSA = find_seizures("CSA", expert_data_list[3])

# Combine all DataFrames
seizures_df = pd.concat(
    [seizures_df_expert_A, seizures_df_expert_B, seizures_df_expert_C, seizures_df_expert_CSA],
    axis=0,
    ignore_index=True
)

# Save to CSV
seizures_df.to_csv("all_expert_seizure_chunks.csv", index=False)
