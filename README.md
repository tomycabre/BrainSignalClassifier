# Brain Signal Classifier

by Tomas Cabrera.

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project analyzes EEG signals to classify different brain activities. It focuses on detecting changes when a subject listens to music (Beethoven and death metal) compared to a baseline state.

## Data Description

The EEG dataset includes recordings from a single subject under different conditions:

- **Baseline**: The subject is at rest; used as negative examples.
- **Beethoven**: EEG while listening to Beethoven's music.
- **Deathmetal**: EEG while listening to death metal music.

### Data Files

The data files are located in the `data` directory:

- `baseline.dat`
- `beethoven.dat`
- `deathmetal.dat`

Each `.dat` file contains the following columns:

- `timestamp`
- `counter`
- `eeg`
- `attention`
- `meditation`
- `blinking`

### Recording Details

- **Duration**: Each recording is between 10 and 11 minutes long.
- **Sampling Frequency (Fs)**: 512 Hz.

## Features

### Data Loading

- Reads EEG data from `.dat` files using pandas.

### Signal Processing

- **Temporal Smoothing**: Applies a moving average filter to smooth the EEG signals.
- **Spectral Filtering**: Uses a Butterworth bandpass filter to isolate frequencies between 1 Hz and 50 Hz.

### Visualization

- Plots raw, smoothed, and filtered EEG signals for each condition.

### Feature Extraction

- Computes statistical features (mean, standard deviation, skewness, kurtosis) over sliding windows.
- Labels extracted features according to the condition (Beethoven or death metal).

### Classification

- Merges features into a single dataset.
- Splits data into training and testing sets.
- Trains a Random Forest classifier to distinguish between the two conditions.

### Evaluation

- Generates a confusion matrix.
- Calculates accuracy, precision, recall, and F1-score.
- Provides a classification report.

## Installation

### Requirements

- Python +3.11

### Packages:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `seaborn`

### Step 1: Clone the Repository

```bash
git clone https://github.com/tomycabre/BrainSignalClassifier.git
cd AI-Web-Scrapper
```

### Step 2: Virtual Environment Setup

#### Windows

```bash
py -m venv venv
```

#### MacOS

```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment

#### Windows

```bash
.\venv\Scripts\activate.bat
```

#### MacOS

```bash
source venv/bin/activate
```

### Step 4: Install Required Packages

#### Windows

```bash
py -m pip install -r requirements.txt
```

#### MacOS

```bash
pip install -r requirements.txt
```

## Usage

### Prepare Data:

Ensure that the EEG data files (`baseline.dat`, `beethoven.dat`, `deathmetal.dat`) are placed in the `data` directory.

### Run the Main Script:

```bash
python main.py
```

### Process Overview:

- **Evaluation**:
  - Displays a confusion matrix.
  - Prints accuracy and classification report.

## Results

### Evaluation Metrics

- **Accuracy**: Proportion of correct predictions over total predictions.
- **Precision**: True positives divided by the sum of true and false positives.
- **Recall**: True positives divided by the sum of true positives and false negatives.
- **F1-Score**: Harmonic mean of precision and recall.

### Sample Output

```markdown
Reporte de Clasificación:
precision recall f1-score support

           0       0.99      0.99      0.99     15234
           1       0.99      0.99      0.99     14871

    accuracy                           0.99     30105

macro avg 0.99 0.99 0.99 30105
weighted avg 0.99 0.99 0.99 30105
```

### Plots

- **Raw Signals**: Visual representation of the original EEG data.
- **Smoothed Signals**: EEG data after applying the moving average filter.
- **Filtered Signals**: EEG data after bandpass filtering.
- **Confusion Matrix**: Heatmap showing the classifier's performance.

## Contributing

We welcome contributions! Please open issues or submit pull requests.

### How to Contribute

1. **Fork the Repository**:
   - Click the 'Fork' button on the repository page.
2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/tomycabre/BrainSignalClassifier.git
   cd brain-signal-classifier
   ```

3. **Create a New Branch**:

   ```bash
   git checkout -b feature/your_feature_name
   ```

4. **Make Changes**: Implement your feature or fix.

5. **Commit Changes**:

   ```bash
   git commit -am 'Add new feature'
   ```

6. **Push to Your Fork**:

   ```bash
   git push origin feature/your_feature_name
   ```

7. **Submit a Pull Request**: Go to the original repository and create a new pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
