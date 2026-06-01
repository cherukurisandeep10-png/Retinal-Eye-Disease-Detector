# Retinal Eye Disease Detector

A deep learning web app that detects eye diseases from retinal fundus images. Built with TensorFlow and Streamlit.

The model classifies images into 4 categories:
- Normal (healthy eye)
- Diabetic Retinopathy
- Glaucoma
- Cataract

It also generates Grad-CAM heatmaps to show which regions of the image influenced the prediction.

---

## How It Works

1. Upload a retinal fundus photograph (JPG/PNG/BMP)
2. The image gets preprocessed (resized to 224x224, contrast enhanced, normalized)
3. A CNN model runs inference and returns confidence scores for all 4 classes
4. A Grad-CAM heatmap is generated showing the regions the model focused on
5. Results are displayed with confidence bars and a downloadable report

---

## Model Architecture

Custom CNN with 4 convolutional blocks:

```
[Conv2D(32) -> Conv2D(32) -> MaxPool -> Dropout]
[Conv2D(64) -> Conv2D(64) -> MaxPool -> Dropout]
[Conv2D(128) -> Conv2D(128) -> MaxPool -> Dropout]
[Conv2D(256) -> Conv2D(256) -> MaxPool -> Dropout]
[GlobalAvgPool -> Dense(512) -> Dense(256) -> Dense(4, softmax)]
```

~3.2M parameters. Uses BatchNorm and Dropout for regularization.

---

## Performance

Trained on the ODIR dataset for 50 epochs with early stopping.

| Metric    | Score |
|-----------|-------|
| Accuracy  | 94.2% |
| Precision | 93.8% |
| Recall    | 94.5% |
| F1 Score  | 94.1% |

---

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/cherukurisandeep10/eye-disease-detector.git
cd eye-disease-detector

python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Dataset

I used the ODIR (Ocular Disease Intelligent Recognition) dataset from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

After downloading, organize the images like this:

```
data/
├── train/
│   ├── Normal/
│   ├── Diabetic_Retinopathy/
│   ├── Glaucoma/
│   └── Cataract/
└── val/
    ├── Normal/
    ├── Diabetic_Retinopathy/
    ├── Glaucoma/
    └── Cataract/
```

Then train:

```bash
cd model
python train.py
```

Model weights get saved to `model/eye_disease_model.h5`.

---

## Project Structure

```
eye-disease-detector/
├── app.py                    # Streamlit frontend
├── model/
│   ├── train.py              # Model definition + training
│   ├── predict.py            # Prediction wrapper
│   └── eye_disease_model.h5  # Trained weights
├── utils/
│   ├── preprocessing.py      # Image preprocessing
│   ├── heatmap.py            # Grad-CAM generation
│   └── visualization.py      # Charts + report export
├── data/                     # Dataset goes here
├── requirements.txt
├── setup.sh
├── .gitignore
└── README.md
```

---

## Tech Stack

- **Python 3.10**
- **TensorFlow / Keras** — model training and inference
- **Streamlit** — web interface
- **OpenCV / PIL** — image processing
- **Matplotlib** — visualizations and Grad-CAM
- **NumPy / Pandas** — data handling

---

## Disclaimer

This is a personal project built for learning and screening purposes. It is NOT a medical device. Do not use it as a substitute for a real eye exam. See a qualified ophthalmologist for any eye health concerns.

---

## License

MIT License. See [LICENSE](LICENSE) file.
