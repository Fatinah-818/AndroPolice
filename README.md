# AndroPolice

AndroPolice is a powerful tool designed for analyzing Android APK files to detect potential malware. Leveraging machine learning, static and dynamic analysis techniques, AndroPolice extracts detailed information from APKs, evaluates their permissions, and provides a classification as either **Benign** or **Malware**. It also highlights dangerous permissions and offers insights into the APK's behavior.

---

## Features

- **APK Analysis**: Perform detailed static and dynamic analysis of APK files.
- **Feature Extraction**: Extract critical features such as permissions, API calls, and more.
- **Malware Detection**: Classify APKs as **Benign** or **Malware** using a machine learning model.
- **Dangerous Permissions**: Identify and list the top dangerous permissions used by the APK.
- **Machine Learning**: Uses a robust Random Forest classifier trained on a comprehensive dataset.
- **Report Generation**: Saves extracted data and analysis results in CSV format for further review.

---

## Installation

### Prerequisites

- **Python 3.8 or later**
- **Docker**: Ensure Docker is installed and running. [Download Docker](https://www.docker.com/)
- Required Python libraries:
  - `scikit-learn`
  - `xgboost`
  - `pandas`
  - `numpy`
  - `joblib`
  - `imblearn`

Install the required libraries:
```bash
pip install -r requirements.txt
