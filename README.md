# Detecting-Pixelated-Image-Correcting-it
This project extracts features from image patches and trains a stacking classifier for classification tasks.

### Part 1: Pixelation Detection 
#### Overview
The script processes images by dividing them into patches and extracting various features from each patch. The features include Edge detection using canny edge Detection, Block-Like patterns using Discrete Cosine Transform (DCT), Histogram of pixel intensities, Histogram of Oriented Gradients (HOG) for Texture Analysis, and Frequency Analysis using Fast Fourier Transform (FFT).

#### Feature Extraction
##### Extract Patch Features
Edge Detection: Uses the Canny edge detector to compute edge density.
Block-like Patterns using DCT: Computes the Discrete Cosine Transform of the patch and extracts the first 8x8 coefficients.
Histogram of Pixel Intensities: Calculates and normalizes the histogram of pixel values.
HOG for Texture: Computes Histogram of Oriented Gradients features.
Frequency Analysis using FFT: Computes the Fast Fourier Transform and extracts the first 8x8 coefficients.
```Python
def extract_patch_features(patch):
    # Code for feature extraction
```

##### Process Each Patch
Divides the image into patches and processes each patch using the extract_patch_features function.
```Python
def process_patch(img, y, x, patch_size):
    # Code to process each patch
```

##### Extract Image Features
Extracts features from the entire image by dividing it into patches and aggregating the features.
```Python
def extract_image_features(image_path, patch_size=128, stride=32):
    # Code to extract features from the image
```
##### Stacking Classifier 
The script trains a stacking classifier using different machine learning models, including Random Forest, XGBoost, and Histogram-based Gradient Boosting. 
The final estimator is a Multi-Layer Perceptron (MLP).
```Python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

estimators = [
    ('rf', RandomForestClassifier(...)),
    ('xgb', XGBClassifier(...)),
    ('hclf', HistGradientBoostingClassifier(...))
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=MLPClassifier(random_state=42),
    cv=5
)
```
##### Training and Evaluation
Trains the stacking classifier on the training dataset and evaluates its performance on the test dataset.
```Python
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Stacking Classifier: {accuracy:.2f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))
```
