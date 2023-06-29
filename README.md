
# Gesture Recognition using Wearable Devices

## Author

- [Rahil Shah](https://github.com/rahilshah17)


## Introduction

This project on gesture recognition using wearable devices was conducted under the guidance of Anil Prabahakar. It was undertaken as a part of my undergraduate research credits. The project explored the exciting and rapidly advancing field of gesture recognition using wearable devices, which has the potential to revolutionize how we interact with electronic devices and assist individuals with disabilities.

Wearable devices, such as smartwatches, fitness trackers, and gloves, are equipped with sensors that capture data about the wearer's movements. This data can be leveraged to develop algorithms capable of recognizing various types of gestures, including waving, pointing, and tapping.

The primary objective of this project was to investigate different methods and technologies employed in gesture recognition using wearable devices. Additionally, the project aimed to develop a prototype system capable of accurately recognizing a predefined set of gestures. The project involved collecting data from users performing the gestures and employing machine learning algorithms to train and test the gesture recognition system. The system's accuracy and usability were evaluated, and the results were presented and discussed.

By successfully developing a functional prototype of a gesture recognition system using wearable devices, this project contributed to the growing body of research in this field. It showcased the potential of this technology for practical applications and provided valuable insights for future research and development efforts in the domain of human-computer interaction.
## Wearable devices for gesture recognition

Wearable devices such as smartwatches, fitness
trackers, and gloves have become increasingly popular
in recent years, and they are also being
used for gesture recognition. These devices are
equipped with various sensors that allow them
to capture data about the wearer’s movements,
which can then be analyzed to recognize various
gestures.

Smartwatches and fitness trackers, for example,
often have accelerometers, gyroscopes, and
magnetometers that can detect changes in the
wearer’s orientation, acceleration, and magnetic
field. These sensors can be used to capture data
about the wearer’s arm and hand movements,
which can then be analyzed to recognize various
gestures such as waving, tapping, and swiping.

Smartwatches and fitness trackers can be used to
monitor and track the movements of people with
disabilities during therapy and exercises. By analyzing
the data collected from the sensors on these
devices, therapists and doctors can gain insights
into the patient’s movements and provide more
personalized and effective treatments. For example,
a smartwatch can be used to monitor the
movements of a person with Parkinson’s disease
during exercises designed to improve their balance
and gait, and provide feedback to the patient and
therapist on their progress.

Overall, wearable devices for gesture recognition
have the potential to transform the way that people
with disabilities interact with the world around
them and receive medical treatment. By providing
more accurate and personalized feedback on
their movements, these devices can help people
with disabilities to achieve greater independence
and improve their quality of life.
## Data collection

To collect data for this project, an Arduino RP
2040 Connect chip was used to capture sensor values
from the wearable device. A custom code was
uploaded to the chip, which allowed it to transmit
the sensor values to a computer via a serial connection.
The code was designed to capture sensor
values from the accelerometer and gyroscope sensors
on the wearable device. The arduino code
that was used to collect the data can be found
[here](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/Arduino_code/Arduino_code.ino). An image of my hand collecting the data is
as shown in the figure 1.

![App Screenshot](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/images/photo_2023-04-04_17-23-08.jpg)

To label the data, a set of predefined gestures were
performed one by one and the corresponding sensor
values were recorded and saved to a CSV file.
Each gesture was repeated multiple times to ensure
a sufficient amount of data was collected. The
collected data was then labeled according to the
corresponding gesture that was performed. The
gestures include right swing about wrist, left swing
about wrist, flexion and extension, adduction and
abduction motions. The figure 2 suggests the
range of gesture that were used. The link to the
data that was collected can be found [here](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/tree/main/data_collected). 

![App Screenshot](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/images/tns9uhud%20(1).png)

The data collection process allowed for the creation
of a labeled dataset, which was used to
train and test the gesture recognition model. The
dataset consisted of a set of sensor values for each
gesture, which was used to train the model to recognize
the specific gesture. Further 80% of the
data was used for training purposes and 20% of
the data was used as testing.
## Preproceesing of the data

After collecting the data, the next step was to preprocess
the data in order to prepare it for training
the gesture recognition model. The raw data consisted
of a set of sensor values for each gesture,
with each sensor value corresponding to a specific
point in time.

Analysis of the data revealed that each gesture
took an average of around 1 second to be performed.
In order to reduce the dimensionality
of the data, 10 consistent samples were taken at
0.1 second intervals for each gesture. These 10
samples were then combined to form a single gesture,
resulting in a data point consisting of 10 time
stamps and corresponding sensor values.

To represent the data in a format suitable for
training the model, the data was preprocessed into
a 3-dimensional array. The first dimension represented
the number of gestures in the dataset,
the second dimension represented the number of
time stamps per gesture (in this case, 10), and the
third dimension represented the sensor values for
each time stamp. This resulted in a dataset of 3-
dimensional arrays, where each array represented
a single gesture.

The preprocessing step was crucial in reducing the
dimensionality of the data and making it suitable
for training the gesture recognition model. By
representing each gesture as a single data point
in a 3-dimensional array, the model could learn
to recognize the specific patterns of sensor values
associated with each gesture.

Overall, the data preprocessing step allowed for
the creation of a labeled dataset that was suitable
for training the gesture recognition model.
The resulting dataset consisted of 3-dimensional
arrays, where each array represented a single gesture
and contained the sensor values for each time
stamp.
## Machine learning algorithm

After preprocessing the data, I applied a machine
learning algorithm to classify the gestures.
Specifically, I used a Convolutional Neural Network
(CNN) with one-dimensional convolutional
layers. The CNN was implemented in Python using
the Keras API with a TensorFlow backend.
The CNN architecture consisted of two convolutional
layers with 64 filters each, followed by a
dropout layer to avoid overfitting, and a max pooling
layer to reduce the spatial dimensions of the
feature maps. The output of the max pooling layer
was flattened and fed to two fully connected layers
with 100 and the number of classes (5) units,
respectively. The final layer used the softmax activation
function to produce the class probabilities.
The model was trained using the categorical
cross-entropy loss function and the Adam optimizer.
The model used is represented as below.

![App Screenshot](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/images/model%20(1).png)

To evaluate the performance of the model, I used
a 20-fold cross-validation approach. For each fold,
I trained the model on 80% of the preprocessed
data and evaluated it on the remaining 20%. I repeated
the experiment 10 times and calculated the
mean and standard deviation of the classification
accuracy. Additionally, I calculated the confusion
matrix to visualize the classification results. The
full code can be accessed from [here](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/Machine_learning_wearable_devices.ipynb).
## Results

Over here we summarize the results that we recieved after using the supervised machine leraning approach.

We trained the data on four different machine
learning models with varying numbers of epochs
while keeping the architecture the same. Our results
showed a mean accuracy of 98.9%. Additionally, we plotted
the confusion matrix, which revealed that the
majority of the predictions were correct, with only
a small number of false positives and false negatives.
This high level of accuracy suggests that
our model is effective for the given task and could
be used for future predictions with a high degree
of confidence. The confusion matrix is as given
below


![App Screenshot](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/images/conf1.png)

Here, the labels 0-4 indicate the following gestures:

0 - No motion of the arm

1 - Left swing over the wrist

2 - Right swing over the wrist

3 - Adduction and abduction over shoulders

4 - Flexion over shoulders  
## Further work

In this section, we present the approach used for
analyzing the headband data and grouping similar
gestures together. The goal was to gain insights
into the patterns and variations in head movements
captured by the gyrometer and accelerometer
data using an unsupervised learning approach.


### Headband Data Description


The headband data consisted of time series measurements
from the gyrometer and accelerometer
sensors. Each measurement captured the rotational
and linear movements of the head during
the performance of various gestures. The data was
collected from a group of participants with diverse
backgrounds.

The time series data looked like the image shown
below

![App Screenshot](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/images/1.png)

![App Screenshot](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/images/a.png)

Over here the upper peaks in the "gz" curve represent
an upward motion of head the slight variations
represent no motion while downward peak
represent the downward head motion.

### Unsupervised learning approach

To analyze the headband data, we employed an
unsupervised learning approach that leveraged the
dynamic time warping (DTW) algorithm and kmeans
clustering. The use of unsupervised learning
allowed us to explore the data without relying
on prior knowledge or labeled examples.

#### Dynamic Time Warping (DTW)

We utilized the DTW algorithm to measure the
similarity between pairs of time series data. DTW
is well-suited for comparing sequences with slight
temporal variations, making it ideal for capturing
the nuances in head movements. By calculating
the DTW similarity scores between each pair of
time series, we could identify gestures that exhibited
similar patterns.

#### K-means Clustering

To refine the grouping of gestures, we applied the
k-means clustering algorithm. K-means clustering
partitions the data into a predetermined number
of clusters, where each cluster represents a distinct
group. By assigning each gesture to its corresponding
cluster based on the DTW similarity
scores, we were able to effectively group similar
gestures together.

#### Hierarchical Clustering

In the hierarchical clustering approach, we performed
clustering on a subset of the time series
data (100-200 samples each) to identify similar
gestures in order to reduce computational complexity
and find better results for the data. Using
the ”single” linkage method and Euclidean distance
metric, we obtained a linkage matrix that
represented the hierarchical structure of the clusters.
By assigning cluster labels using the ”maxclust”
criterion and specifying 3 clusters, we labeled
the data points. The cluster labels [50, 100,
150] were used for better visualization. Plotting
the data with these labels revealed distinct clusters,
indicating successful grouping of similar gestures
based on their time series characteristics.
This demonstrates the potential of hierarchical
clustering for gesture recognition tasks, but further
analysis and optimization are necessary to
refine the clustering approach and improve accuracy.

#### Results of unsupervised learning

In this section, we present the results of our gesture
recognition analysis using hierarchical clustering
on the time series data. We compared
the performance of hierarchical clustering and kmeans
clustering algorithms. Our findings revealed
that hierarchical clustering outperformed
k-means clustering in accurately grouping similar
gestures based on their time series characteristics.

Below figures show the clustering results
obtained from hierarchical clustering. These plots
visually demonstrate the success of the clustering
algorithm in capturing distinct patterns and
separating different gestures. The clusters obtained
from hierarchical clustering exhibit clear
boundaries, indicating the effectiveness of the approach.

Our analysis confirms that hierarchical clustering
is a promising technique for gesture recognition in
time series data. The ability to accurately group
similar gestures lays the foundation for developing
robust gesture recognition systems. Further
analysis and refinement of the clustering approach
can enhance the accuracy and efficiency of the system.

The full code for the unsupervised learning approach
can be accessed from [here](https://github.com/rahilshah17/Gesture-recognition-using-wearable-devices/blob/main/Machine_learning_over_headband_data.ipynb).
## Conclusion

In conclusion, this project aimed to develop a gesture
recognition system to aid disabled individuals
in performing daily tasks. The system was built
using gyroscope and accelerometer data from a
wearable device and trained using a deep learning
approach. The results showed that the system
achieved a high accuracy of 98.9%, indicating its
potential as an assistive technology for individuals
with physical disabilities.

Overall, this project demonstrated the potential
of deep learning and mobile sensing technology to
make a positive impact on the lives of individuals
with disabilities. The developed system has the
potential to make tasks that were once challenging
or impossible for disabled individuals, more
accessible and achievable. As such, it could lead
to greater independence and improved quality of
life for those with disabilities, ultimately fulfilling
the project’s main goal.

In addition, the application of the unsupervised
learning approach using dynamic time warping
(DTW) and k-means clustering proved to be a
valuable method for analyzing and grouping similar gestures within the headband data. This unsupervised
approach allowed for the exploration of
hidden patterns and structures within the data,
without relying on prior knowledge or labeled
data. By employing DTW to measure the similarity
between time series data and utilizing k-means
clustering to partition the gestures into distinct
groups, we gained insights into the underlying relationships
and similarities between different head
movements. This unsupervised learning approach
provides a foundation for further analysis and exploration
of the dataset, enabling a better understanding
of the variations and patterns in head
movements.


## Further research opportunities

One of the main limitations of the current model
is its dependency on data collected in a specific
format or manner. However, in today’s era of
abundant data, there is a vast amount of scattered
or unstructured data available that remains
untapped. Therefore, a promising direction for future
research is the development of a model that
can automatically learn from such diverse and heterogeneous
data sources.

By incorporating techniques from unsupervised
and self-supervised learning, as well as leveraging
advancements in deep learning and artificial
intelligence, it is possible to create a more adaptable
and flexible model. This model should be
capable of extracting meaningful patterns and insights
from unstructured data, thereby reducing
the reliance on manually curated or preprocessed
datasets.

Furthermore, research efforts can be directed towards
exploring novel approaches such as transfer
learning, domain adaptation, or data fusion
techniques to enhance the model’s ability to generalize
and perform well on diverse data sources.
This would enable the model to learn from a wide
range of data, including real-world scenarios and
complex environments.

Additionally, the development of techniques for
data preprocessing, feature engineering, and data
augmentation specifically tailored to scattered or
unstructured data can greatly enhance the performance
and applicability of the model. These techniques
should address challenges such as missing
data, data quality issues, and data heterogeneity, ensuring robust and reliable results.

Overall, the exploration of automated learning
models that can effectively utilize scattered or unstructured
data presents a significant avenue for
future research. By overcoming the limitations of
the current model and harnessing the power of
diverse data sources, we can unlock new possibilities
and insights, ultimately advancing our ability
to analyze and understand complex phenomena.
## References

1. The architecture for the training of the data
was inspired from
author = Jason Brownlee,
title = CNN Models for Human Activity
Recognition,
year = 2019,
The link to his work is given [here](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/).

2. The unsupervised learning technique approach
has been inspired from
author = Denyse,
title = Time Series Clustering,
year = 2021,
The link to his work is given [here](https://towardsdatascience.com/time-series-clustering-deriving-trends-and-archetypes-from-sequential-data-bb87783312b4).

3. The algorithm for K-means clustering approach
has been inspired from
author = Alexandra Amidon,
title = How to Apply K-means Clustering to
Time Series Data,
year = 2020,
The link to her work is given [here](https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3).

4. The algorithm for hierarchical clustering
approach has been inspired from
author = Alexandra Amidon,
title = How to Apply Hierarchical Clustering
to Time Series,
year = 2020,
The link to her work is given [here](https://towardsdatascience.com/how-to-apply-hierarchical-clustering-to-time-series-a5fe2a7d8447).