# Quick Start Guide

Get started with the Pneumonia Classification project in minutes!

## 🚀 Setup (5 minutes)

1. **Clone and navigate to project:**
```bash
cd pneumonia_classifier_project
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🏃 Run Training (10-15 minutes)

Train the model with default settings:
```bash
python train.py
```

The script will:
- Download the PneumoniaMNIST dataset automatically
- Train for 5 epochs
- Save model checkpoints in `models/` directory
- Save training history in `results/` directory
- Log progress to console and `logs/training.log`

## 📊 View Results

After training, check:
- **Model checkpoints**: `models/checkpoint_epoch_*.pth`
- **Final model**: `models/final_model.pth`
- **Training history**: `results/training_history.json`
- **Logs**: `logs/training.log`

## 🔍 Run Inference

Classify a chest X-ray image:
```bash
python inference.py --image path/to/image.png --model models/final_model.pth
```

## ⚙️ Customize Configuration

Edit `config.yaml` to change:
- Batch size
- Learning rate
- Number of epochs
- Model architecture
- And more!

## 📝 Example Workflow

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train
python train.py

# 3. Test inference (after training)
python inference.py \
    --image test_image.png \
    --model models/final_model.pth \
    --threshold 0.10
```

## 🐛 Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in `config.yaml`

**Issue**: Module not found
- **Solution**: Make sure virtual environment is activated and dependencies are installed

**Issue**: Dataset download fails
- **Solution**: Check internet connection or manually download from MedMNIST

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [IMPROVEMENTS.md](IMPROVEMENTS.md) to see what was improved
- Explore the code in `src/` directory
- Customize `config.yaml` for your needs

Happy training! 🎉

