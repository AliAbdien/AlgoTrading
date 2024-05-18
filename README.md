
# AlgoTrading

A repository for algorithmic trading strategies and models focused on cryptocurrency.

## Getting Started

### Setting Up the Development Environment

1. **Clone the Repository**:
```bash
git clone https://github.com/your_username/AlgoTrading.git
cd AlgoTrading
```

2. **Set Up the Conda Environment**:
Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed. Once installed, you can set up the environment using:
```bash
conda env create -f AlgoTradingEnv.yml
```
Activate the environment:
```bash
conda activate AlgoTradingEnv
```

3. **Run the Notebooks/Scripts**:
Once the environment is set up and activated, you can run the provided Jupyter notebooks or scripts.

### Repository Structure

- `AlgoTradingEnv.yml`: Conda environment configuration file.
- `data/`: Directory containing all data files.
  - `processed/`: Processed data ready for modeling.
  - `raw/`: Raw data files.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `scripts/`: Standalone scripts.
- `src/`: Source code for the project.
  - `data/`: Data processing scripts.
  - `features/`: Feature engineering scripts.
  - `models/`: Model training and evaluation scripts.
- `tests/`: Test scripts to ensure code quality and correctness.

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE.md` file for details.


## Setup & Installation

For team members looking to get the codebase and environment set up on their machines, we provide two methods: using Conda or using pip. 

### Using Conda (Recommended)

1. **Clone the Repository**
    ```
    git clone https://github.com/your_username/AlgoTrading.git
    cd AlgoTrading
    ```

2. **Create a Conda Environment**
    ```
    conda env create -f AlgoTradingEnv.yml
    ```

3. **Activate the Conda Environment**
    ```
    conda activate tradingEnv
    ```

4. **Run your Python scripts or Jupyter Notebooks within this environment**

### Using pip

1. **Clone the Repository**
    ```
    git clone https://github.com/your_username/AlgoTrading.git
    cd AlgoTrading
    ```

2. **(Optional) Create a Virtual Environment**
    ```
    python -m venv venv_name
    source venv_name/bin/activate  # On Windows, use: .env_name\Scriptsctivate
    ```

3. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Run your Python scripts or Jupyter Notebooks within this environment**

**Note**: When using pip, ensure you're using a Python version compatible with the project (as mentioned in the `AlgoTradingEnv.yml` file).

