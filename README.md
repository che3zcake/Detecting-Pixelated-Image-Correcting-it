# Detecting-Pixelated-Image-Correcting-it
This project is to build two models: one that detects pixelation in an image and one that resolves the pixelation and we combine these two models to work together.

***The [PPT]() regarding our entire project can be found in the repository***

### Part 0: How to Use
1. Open a GitBash terminal and clone the directory and change the directory to Detecting-Pixelated-Image-Correcting-it
```bash
git clone https://github.com/che3zcake/Detecting-Pixelated-Image-Correcting-it.git
```
```bash
cd Detecting-Pixelated-Image-Correcting-it
```
2. Open tenma0.ipynb and set the path for detection model ***pixelation_model.pkl** and correction model ***generator.pth*** to path
```python
model_path = os.path.join(r'path_of_pixelation_model.pkl_file', 'pixelation_model.pkl')
```
```python
#Example Usage
model_path = os.path.join(r'C:\Users\nilot\OneDrive\Desktop\IntelAi', 'pixelation_model.pkl')
```
```python
if not torch.cuda.is_available():
    generator.load_state_dict(torch.load(r'path_of_generator.pth_file', map_location=torch.device('cpu')))
else:
    generator.load_state_dict(torch.load(r'path_of_generator.pth_file'))
```
```python
#Example Usage
if not torch.cuda.is_available():
    generator.load_state_dict(torch.load(r'C:\Users\nilot\OneDrive\Desktop\IntelAi\generator.pth', map_location=torch.device('cpu')))
else:
    generator.load_state_dict(torch.load(r'C:\Users\nilot\OneDrive\Desktop\IntelAi\generator.pth'))
```
3. Give the path to your image and run the cell
```python
usage(image_path = r'image_path')
```              
4. ***OPTIONAL*** Install the libraries with exact versions we used to test and train our model
```python
pip install -r requirements.txt
```

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
### Part 2: Correction Model 
#### Overview
Our SRGAN model is a neural network designed to correct images with pixelation to give an output of the same image without pixelation artifact. 
The model consists of two main components: the Generator and the Discriminator.

#### 1. Generator Network:
The Generator aims to generate high-resolution (HR) images from low-resolution (LR) images. The network architecture includes:

##### Initial Convolutional Layer:
A convolutional layer with 64 filters and a kernel size of 9x9, followed by a Parametric ReLU (PReLU) activation function. This layer processes the input LR image.

##### Residual Blocks:
The core of the generator consists of several residual blocks (in this case, 16 blocks). Each block contains:
Two convolutional layers with 64 filters and a kernel size of 3x3.
Batch normalization layers.
PReLU activation.
These residual blocks help in learning the identity mapping, which is crucial for generating detailed high-resolution images.

##### Upsampling Layers:
Two upsampling layers using PixelShuffle to increase the resolution of the feature maps. Each layer consists of:
A convolutional layer with 256 filters and a kernel size of 3x3.
PixelShuffle operation to increase the spatial resolution by a factor of 2.
PReLU activation.

##### Final Convolutional Layer:
A convolutional layer with 3 filters and a kernel size of 9x9 to produce the final high-resolution image.
The generator is trained to minimize the Mean Squared Error (MSE) between the generated image and the ground truth high-resolution image, as well as the perceptual loss calculated using a pre-trained VGG network to ensure the generated image looks perceptually realistic.

#### 2. Discriminator Network:
The Discriminator's role is to distinguish between real high-resolution images and the generated images from the generator. The architecture includes:

##### Convolutional Layers:
Multiple convolutional layers with increasing filter sizes (64, 128, 256, 512) and kernel sizes of 3x3.
Leaky ReLU activations after each convolutional layer.
Batch normalization to stabilize the training process.
Downsampling using strided convolutions.

##### Fully Connected Layers:
A fully connected layer with 1024 units and Leaky ReLU activation.
A final fully connected layer with a single output unit followed by a Sigmoid activation to output the probability of the input image being real or fake.
The discriminator is trained to minimize the Binary Cross-Entropy (BCE) loss between the predicted probabilities and the true labels (real or fake).

## Architecture
 <p align="center">
  <b>Generator</b>
</p>
<p align="center">
  <img src="Images/gen.png" width="100%" />
</p>

 <p align="center">
  <b>Discriminator</b>
</p>
<p align="center">
  <img src="Images/dis.png" width="100%" />
</p>

## :star: Results
###  <ins>Detection Results: </ins>

**Datasets used for testing:**

- Div2K (Full dataset - 900 images)
- Flickr2K (Test split - 284 images)

  <p align="center">
  <b>Non Pixelated</b> | <b>Pixelated</b> 
</p>
<p align="center">
  <img src="Images/0801np.png" width="400" />
  <img src="Images/0801.png" width="400" />
</p>

**Performance:**

| Metric           | Proposed Method on Flickr2K +Div2k | 
|------------------|------------------------------------|
| **Precision**    | 0.99                               | 
| **Recall**       | 0.99                               | 
| **F1 Score**     | 0.99                               | 
| **Accuracy**     | 0.99                               |
| **FPS**          | 300                                |
| **Model Size**   | 17.2 MB                            | 


###  <ins>Correction Results: </ins>
#### Metrics are calculated on set5 images 

All the metrics calculations can be found in the [matrics.ipynb](https://github.com/che3zcake/Detecting-Pixelated-Image-Correcting-it/blob/main/metrics.ipynb) file 


| Metric            | Proposed Model          | SRGAN                    | EDSR                     | FSRCNN                   |
|-------------------|-------------------------|--------------------------|--------------------------|--------------------------|
| PSNR              | 26.556 dB               | 29.99 dB                 | 31.78 dB                 | 30.52 dB                 |
| SSIM              | 0.7690                  | 0.8176                   | 0.8895                   | 0.8548                   |
| LPIPS             | 0.3064                  | 0.1118                   | 0.1922                   | 0.2013                   |
| Speed (FPS)       | 79                      | 12                       | 16                       | 188                      |
| Model Size (MB)   | 6.01                    | 5.874                    | 5.789                    | 0.049                    |

***Sample Output***

<p align="center">
  <b>Low Resolution Images</b> | <b>Super Resolved Images</b>  |<b>High Resolutio Images</b> 
</p>
<p align="center">
  <img src="Images/1.jpg" width="100%" />
</p>

<p align="center">
  <img src="Images/2.jpg" width="100%" />
</p>

<p align="center">
    <img src="Images/3.jpg" width="3000%" />
</p>

<p align="center">
    <img src="Images/4.jpeg" width="100%" />
</p>

### Conclusion
The implementation provided combines two key functionalities: detecting pixelation in an image and enhancing its resolution using a Super-Resolution Generative Adversarial Network (SRGAN).
Key Components:
Pixelation Detection:
This part of the code uses a machine learning model to detect if an image is pixelated or not. It involves:
Extracting various features from image patches, such as edge detection, DCT features, histogram of pixel intensities, HOG for texture, and FFT for frequency analysis.
Aggregating these features to form a comprehensive feature vector for the entire image.
Using a GAN model for Super Resolution:
If the image is found to be pixelated by the detection model, the GAN model is used to enhance its resolution and de-pixelate itThe generator model consists of an initial convolutional layer, multiple residual blocks, upsampling layers using PixelShuffle, and a final convolutional layer.The generated image is then denormalized, saved, and displayed using matplotlib.
This model stands out for its exceptional efficiency and lightweight design, with a combined size of only 30 MB for both detection and correction components. This makes it ideal for embedded systems, unlike many pre-existing models that, while occasionally offering superior results, are often excessively large and memory inefficient.

### References
<p align="left">
    (https://github.com/cszn/DnCNN)
</p>
<p align="left">
    (https://github.com/cszn/FFDNet)
</p>
<p align="left">
    (https://paperswithcode.com/task/image-deblocking/latestN)
</p>
<p align="left">
    (https://github.com/zeyuxiao1997/EDPN/tree/main)
</p>
<p align="left">
    ([https://github.com/cszn/DnCNN](https://github.com/HasnainRaz/Fast-SRGAN))
</p>








