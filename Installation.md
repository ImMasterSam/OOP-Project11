# 🛠️ Installation Manual 

## 🏋️ Gymnasium Base Module
To run Gymnasium environments, follow these steps to set up your development environment:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Navigate to the Gymnasium directory
cd group_project/Gymnasium

# 4. Install Gymnasium in editable mode
pip install -e .

# 5. Install additional dependencies
pip install "gymnasium[classic_control]"
pip install matplotlib
```

## 📦 Additional Module
Different modules are needed for different parts.  
You can install the required modules as follows:

> ⚠️ Make sure to install the required modules in the **Python Virtual Environment**.  

### 🏔️ Part 1: Mountain Car
No additional modules are required.

### 🧊 Part 2: Frozen Lake
Require the `tqdm` and `optuna` module:
```bash
pip install tqdm optuna
```

### 🏎️ Part 3: Car Racing
Require the `box2d` module from *Gymnasium*:

```bash
# Linux/MacOS
pip install "gymnasium[box2d]"

# Windows
pip install "ufal.pybox2d"
```

Require the **Pytorch** module to run neaural networks:

```
pip install torch 
```

> If you have a CUDA-capable GPU and want to utilize it, please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for specific instructions based on your system configuration.

## ⚙️ Environment Settings
```bash
# Linux/MacOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
``` 