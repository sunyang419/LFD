# LFD

## Environment Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate [environment_name]  # Replace with the environment name defined in environment.yml
```

### 2. Install Transformers

```bash
pip install transformers-4.51.3/ # Install the transformers included in the project
```

---
## Run Main Experiment

### 1. Configuration

Edit the following lines in `run_lfd.py`:

```python
# Lines 34-36
parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-chat-hf") # Model name to use
parser.add_argument("--data_path", type=str, default="./data/nq/nq_noise_0.json") # Test data path
parser.add_argument("--output_path", type=str, default="./output/llama2-7b/nq/nq_lfd.json") # Path to save results
```

### 2. Run Inference

```bash
python run_lfd.py
```

### 3. Evaluate Experiment Results

#### 1. Edit `final_eval.py`:
Modify line 7 to set the result directory:

```python
folder = "output/llama2-7b/nq/"
```

#### 2. Run Evaluation Script:

```bash
python final_eval.py
```

#### 3. Result Output Path:

```plaintext
The results will be saved in output/llama2-7b/nq/final_eval.json
```

---

## Performance Evaluation Experiment (Latency, Throughput & Memory Usage)

### 1. Configuration

Edit  `run_ef.py` (similar to the main experiment).

### 2. Run Inference

```bash
python run_ef.py
```

### 3. Evaluate Experiment Results

```bash
python ef_eval.py
```
---
