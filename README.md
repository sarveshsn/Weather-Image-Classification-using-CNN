# Weather-Image-Classification-using-CNN

Weather Image Classification holds significant significance in the realms of meteorology and computer vision, offering far-reaching implications that span a diverse array of activities, including weather forecasting, environmental monitoring, and disaster preparedness. The prevailing techniques predominantly hinge on human interpretations, a factor that can potentially introduce inaccuracies and uncertainties. This underscores the pressing need for more advanced methodologies.

As a solution, the integration of Convolutional Neural Networks (CNNs) into weather image classification has emerged as a robust and pragmatic approach. In this project, our primary objective is to develop an image classifier capable of categorizing diverse weather images. With the aid of this image classifier, predictions about the prevailing weather conditions can be rapidly and accurately made based on input images. This marks a significant advancement over traditional approaches, enhancing both efficiency and accuracy.

## Folder Structure
```
├─ Data_preparation.R
├─ Training.Rmd

README.md

```
## Installing R and RStudio

### 1. Install R:

R is a programming language and software environment used for statistical computing and graphics.

  Visit the official R website: https://cran.r-project.org/
  Click on the "Download R" link at the top of the page.
  Select your operating system (Windows, macOS, or Linux) and download the appropriate installer.
  Run the downloaded installer and follow the on-screen instructions to install R.

### 2. Install RStudio:

RStudio is an integrated development environment (IDE) specifically designed for R.

  Visit the official RStudio website: https://www.rstudio.com/
  Click on the "Products" menu and select "RStudio Desktop."
  Choose the "Free" version, which is sufficient for most users.
  Download the appropriate installer for your operating system (Windows, macOS, or Linux).
  Run the downloaded installer and follow the on-screen instructions to install RStudio.

## Prerequisite

You need to have following packages installed prior to running the code in RStudio. You can run this code in the RStudio script/console. 
```
install.packages("jpeg") #For loading pictures
install.packages("reticulate")  #To integrate Python code and libraries into your R environment if not done exclusively.
install.packages("keras")  #For training CNN model
install.packages("ggplot2) #To plot the learning curves
install.packages("grDevices")  #To Plot accuracy and loss

```
Import the above packages by running the following code - 
```
library(jpeg)
library(reticulate)
library(keras)
library(ggplot2)
library(grDevices)
```

## Dataset

Dataset used in this project can be found at https://www.kaggle.com/datasets/jehanbhathena/weather-dataset?resource=download

Dataset contains labeled 6862 images of different types of weather.

The weather dataset is split into 11 folders, one for each class, with each named after the class of images it contains. i.e. all 'rain' images must be in weather/rain/.

Our original dataset folder structure will look like this.

![filesssnip](https://github.com/ACM40960/project-sarveshsn/assets/93898181/38509c78-edd2-4442-a869-14d56a95994e)


Inside these folder contains images of that particular weather. For example, lightning folder will contain below images.

<img width="647" alt="lightning" src="https://github.com/ACM40960/project-Neha-0994/assets/118282077/eff60ef3-6d39-437d-b140-42eedd3ff49c">


## Data preparation

We must split the dataset into train, validation and test prior training the model. We have split the original dataset into folders like train, test and validation. Each of these 3 folders will contain the 11 different weather class folders. The train, test and validation split is 80-10-10 respectively. 

Running file Data_preprocessing.R will split the folders into train, test and validation folder. Our new dataset will look like this.

![filesssnip2](https://github.com/ACM40960/project-sarveshsn/assets/93898181/4c984259-89fa-4cf0-ba81-4a7d69e73c2c)



## CNN Model

Image classifier is built using Convolutional Neural Network (CNN). We have built a baseline CNN model followed by 2 CNN models with different configurations. Confusion matrix is used to check the precision, recall and F-1 score of these 3 models. These 3 are used as metrics to understand the performance of the model.

### CNN Model 1

•	Architecture: Model 1 (our base model) is CNN with 4 convolution layers and 4 max-pooling layers then followed by 2 fully connected layers. The first convolution layer is set with 32 filters and a 3 × 3 kernel with strides 1 (default). Next 3 convolution layers are set with 64 and 128 filters with 3 × 3 kernels. The fully connected layer uses 512 units and ReLU activation function.

•	Hyperparameters: Learning rate -0.0001 using optimizer - RMSProp

![Model1](https://github.com/ACM40960/project-sarveshsn/assets/93898181/1f3befbe-c769-4aa8-9e32-5481a3802372)


### CNN Model 2

•	Architecture: Four pairs of convolutional layers followed by max-pooling layers.The number of filters increases with depth (64, 128, 256, 512). Additional use of batch normalization and dropout for regularization. Flattening layer followed by two dense (fully connected) layers.

•	Hyperparameters: Learning Rate: 0.001 (using Adam optimizer).	Batch normalization applied after each convolutional and dense layer. Dropout: 0.25 applied after each max-pooling layer and 0.5 after the first dense layer.

![Model2](https://github.com/ACM40960/project-sarveshsn/assets/93898181/476b8cf7-2532-47e8-b298-7aa409eb829f)



### CNN Model 3 

Model 3 is structurally similar to Model 2, but with the added benefit of data augmentation during training. Data augmentation was introduced to increase the model's exposure to a wider variety of training samples. This can lead to improved generalization, as the model learns to recognize important features under various conditions.

## Project Implementation

•	Execute the Data_preprocessing.R script to partition the dataset into distinct train, test, and validation directories.

•	Proceed to execute the Training.Rmd script for the purpose of training the models. It is crucial to ensure the accuracy of the specified path within your code.

Each model is trained for 50 epochs with 100 steps_per_epoch and 50 validation steps. Achieve this by utilizing the provided code.

#### Important note: Running the model will necessitate a substantial amount of time and computational resources.
  ```
# Fit the model
fit <- model %>% fit(
  train_generator,
  steps_per_epoch = 100,
  epochs = 50,
  validation_data = validation_generator,
  validation_steps = 50
)

```

## Result 

Our primary focus lies in determining the most effective Convolutional Neural Network (CNN) model for our multi-class dataset. This assessment revolves around evaluating the performance metrics of each model, specifically considering test set accuracy, along with macro-averaged precision, recall, and F1 scores. These assessments are grounded in the examination of confusion matrices.

Upon conducting a comprehensive comparative study involving three distinct models, we have derived insightful results. The outcomes of this comparative study are visually presented as follows:

![comp study](https://github.com/ACM40960/project-sarveshsn/assets/93898181/9b2704cf-ebbc-4865-af32-5231d0cf8462)


Furthermore, to delve deeper into the evaluation of our most promising model, Model 2, we scrutinized its performance in classifying the 11 distinct classes. This evaluation encompassed an in-depth analysis of performance metrics for each individual class, depicted as follows:

![Best Model](https://github.com/ACM40960/project-sarveshsn/assets/93898181/8bfc15fb-91d2-4195-b9fe-2dd0110aab82)


### Inference - 

Drawing conclusions from our findings, it becomes evident that Model 2 has exhibited exceptional performance across all considered metrics. This suggests that its architectural design and parameter configuration are particularly well-suited for the intricate task of Weather Image Classification. Notably, Model 2's success can be attributed to its strategic implementation of various techniques, including heightened filter counts, batch normalization, dropout mechanisms, and diverse kernel sizes. In contrast, while Model 1 and Model 3 possess their individual merits, they do not match the proficiency displayed by Model 2 in terms of accuracy, precision, recall, and F1 scores. This comprehensive assessment underlines the pivotal role of architecture and configuration in shaping the prowess of CNN models, with Model 2 emerging as the superior choice for our specific Weather Image Classification task.

## Future Scope and Applications

### Future Scope

•	Fine-grained Weather Classification: While our project focuses on broader weather categories, future work could involve fine-grained classification, distinguishing between variations within each category, such as different types of clouds or intensities of rain.

•	Real-time Integration: Implementing real-time integration of the trained CNN model with weather monitoring systems, outdoor cameras, and drones can provide instant weather condition updates for enhanced decision-making.

•	Transfer Learning: Incorporating transfer learning by using pre-trained CNN models like ResNet, VGG, or Inception can leverage their learned features for improved performance with less training data.

•	Multimodal Weather Classification: Expanding the project to classify weather conditions based on multiple input data sources, such as combining visual data from images with textual weather reports, can lead to more accurate predictions.

### Potential Applications

•	Smart Transportation Systems: Incorporate weather prediction models into smart transportation systems to optimize routes, schedules, and vehicle operations based on current and forecasted weather conditions.

•	Agricultural Management: Assist farmers in making informed decisions about crop cultivation, irrigation, and pest control by providing real-time weather insights.

•	Energy Management: Optimize energy consumption and distribution in smart grids by predicting demand fluctuations due to changing weather conditions.

•	Emergency Response: Aid emergency responders and disaster management teams with accurate and timely weather information, enabling them to plan and execute actions effectively.

•	Tourism and Outdoor Activities: Help tourists and outdoor enthusiasts plan activities by providing weather forecasts that consider specific preferences and requirements.

## Acknowledgement

I wish to extend my heartfelt appreciation to Dr. Sarp Akcay for his invaluable guidance and unwavering support throughout the duration of the module at University College Dublin. The knowledge and insights he shared significantly contributed to the success of this project.

My gratitude also extends to University College Dublin for generously providing the essential resources that laid the foundation for the completion of this project. Their commitment to academic excellence has been instrumental in shaping the quality and depth of this endeavor.

## References

1. H. Li, X.J. Wu, and J. Kittler. "2018 24th International Conference on Pattern Recognition (ICPR)", 2018.

2. Phung and Rhee. "A High-Accuracy Model Average Ensemble of Convolutional Neural Networks for Classification of Cloud Image Patches on Small Datasets." Applied Sciences, 9:4500, 2019.

3. Van Hiep Phung and Eun Joo Rhee. "A High-Accuracy Model Average Ensemble of Convolutional Neural Networks for Classification of Cloud Image Patches on Small Datasets." Applied Sciences, 9(21), 2019.

4. Congcong Wang, Pengyu Liu, Kebin Jia, Xiaowei Jia, and Yaoyao Li. "Identification of Weather Phenomena Based on Lightweight Convolutional Neural Networks." Computers, Materials and Continua, 64(3):2043–2055, 2020.

5. Haixia Xiao, Feng Zhang, Zhongping Shen, Kun Wu, and Jinglin Zhang. "Classification of Weather Phenomenon from Images by Using Deep Convolutional Neural Network." Earth and Space Science, 8, 2021.

## Authors

• Sarvesh Naik 

