# OOP - Group Project (9)

## 👥 Group Members
| Name     | Student ID   | Class |
|--------- |------------- |-------|
| 劉育希   | B123040049   | 中文班 `CSE391`  |
| 侯廷翰   | B123040044   | 中文班 `CSE391`  |
| 柯伯諺   | B123245016   | 全英班 `CSE3002` |


## 📄 Project Content
Not yet available.

  
## 🛠️ Installation

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

## 🚀 Running the Project

### 🏔️ Part 1: Mountain Car
Train and test the reinforcement learning agent:

```bash
# Train the agent
python mountain_car.py --train --episodes 5000

# Render and visualize performance
python mountain_car.py --render --episodes 10
```

### 🧊 Part 2: Frozen Lake
Run the Frozen Lake environment:

```bash
python frozen_lake.py
```

### 🏭 Part 3: OOP Project Environment
Execute the custom OOP environment:

```bash
python oop_project_env.py
```

> **Environment Settings**   
> ```bash
> # Linux/MacOS
> source .venv/bin/activate
> 
> # Windows
> .venv\Scripts\activate
> ``` 

## 🤝 Contribution


