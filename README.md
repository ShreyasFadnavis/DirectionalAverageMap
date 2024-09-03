# Directional Average Model for Diffusion MRI

This repository contains a Python implementation of a custom diffusion MRI model called the **Directional Average Model**. This model computes the directional average of diffusion MRI signals across unique b-value shells and fits a linear model to the log-attenuated signals.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Reference](#reference)

## Installation

To run this code, you need Python 3.x and the following dependencies:

- `numpy`
- `dipy`

You can install the required packages using pip:

```bash
pip install numpy dipy
```

## Usage

1. **Fetch the Data:**
   The code fetches the CENIR multiband diffusion MRI dataset and uses it to demonstrate the functionality of the Directional Average Model. 

2. **Run the Model:**
   To run the model, use the following command in your terminal:

   ```bash
   python directional_average_model.py
   ```

   The script will:
   - Fetch the CENIR dataset.
   - Initialize the `DirectionalAverageModel` using the gradient table from the dataset.
   - Fit the model to the data.
   - Print the fitted parameters and the predicted signal attenuation.

## Example

Below is an example of how to run the Directional Average Model script:

```python
# Fetch and read the CENIR dataset
fetch_cenir_multib(with_raw=False)
img, gtab = read_cenir_multib()
data = img.get_fdata()

# Initialize the model
dam = DirectionalAverageModel(gtab=gtab)

# Fit the model
daf = dam.fit(data)

# Example usage of the fitted model
predicted_signal = daf.predict(gtab)
print("Fitted Parameters:", daf.P, daf.V)
print("Predicted Signal:", predicted_signal)
```

## Reference

The implementation of the Directional Average Model is inspired by the paper:

Cheng, Hu, Newman, Sharlene, Afzali, Maryam, Fadnavis, Shreyas Sanjeev, and Garyfallidis, Eleftherios. "Segmentation of the brain using direction-averaged signal of DWI images." *Magnetic Resonance Imaging*, vol. 69, pp. 1-7, 2020. Publisher: Elsevier.
