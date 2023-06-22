# AI-Exam-machine-learning
Machine learning 

# Description of the project
The project aims to explore the application of Artificial Neural Networks on MNIST dataset. The use of hyper-parameters makes the model to be dynamic because of the use of the random search Method from Keras-Tuner, which every time executing or training the model will result in a different model, hyper-parameters, and accuracy. 

# What is ANN
Artificial Neural Networks (ANNs), the cornerstone of Deep Learning, have revolutionized the field of machine learning and artificial intelligence. ANNs are computational models inspired by the biological neural networks in the human brain, designed to learn and perform tasks by recognizing patterns and relationships in vast amounts of data. With their ability to process complex information, ANNs have made significant advancements in various domains, including computer vision, natural language processing, and speech recognition. The availability of large-scale datasets, such as MNIST, coupled with advances in computational power and optimization techniques, has propelled the success of Deep Learning. It has enabled breakthroughs in image classification, object detection, image generation, machine translation, and many other domains. Deep Learning continues to push the boundaries of artificial intelligence, empowering machines to perform tasks that were once considered exclusive to human intelligence.

# What is MNIST dataset
“The MNIST dataset is a widely used benchmark dataset in the field of machine learning and computer vision. MNIST stands for Modified National Institute of Standards and Technology, which refers to the organization that created the dataset. It was developed as a resource for researchers to evaluate and compare the performance of various machine learning algorithms, particularly those related to image recognition and classification. The MNIST dataset consists of a collection of grayscale images of handwritten digits, ranging from 0 to 9. It is composed of two main parts: a training set and a test set. The training set contains 60,000 images, while the test set contains 10,000 images. These images are evenly distributed across the ten different digits, meaning there are roughly 6,000 images for each digit in the training set and 1,000 images for each digit in the test set.
Each image in the MNIST dataset is a 28x28-pixel square, resulting in a total of 784 pixels per image. The grayscale values of these pixels represent the intensity of the corresponding pixel, ranging from 0 (white) to 255 (black). Each image is labeled with the correct digit it represents, allowing researchers to train their machine learning models on the training set and evaluate their performance on the test set. The MNIST dataset has gained popularity due to its simplicity, availability, and its ability to provide a standardized benchmark for evaluating algorithms. It has served as a foundation for developing and testing various image classification and pattern recognition techniques, including traditional machine learning algorithms as well as more advanced deep learning models. The dataset has played a crucial role in advancing the field of computer vision and has become a reference point for researchers and practitioners working in this domain.” 

# Which Libraries Used
The study makes use of Python 3.10 as a language and Google Colab from Google as a tool to develop the ANN model. And the packages used are as follows:
•	Scikit-Learn: it is a popular and free machine learning library for the Python programming language. (Scikit-learn, 2017)
•	Numpy:  it is a powerful Python library that provides a multidimensional array object, along with various derived objects like masked arrays and matrices. (Scikit-learn, 2017)
•	Tensarflow: it is an open-source machine learning framework developed by Google. It is designed to facilitate the development and deployment of machine-learning models
•	Seaborn: it is a Python data visualization library built on top of matplotlib. It provides a higher-level interface for creating aesthetically pleasing and informative statistical graphics.
•	Keras: it is an open-source deep learning framework written in Python. Initially developed as a user-friendly interface to build neural networks on top of other deep learning libraries, Keras has gained widespread popularity due to its simplicity and ease of use.
•	Matplotlib: it is a popular data visualization library for Python. It provides a comprehensive set of tools for creating static, animated, and interactive visualizations in various formats, such as line plots, scatter plots, bar plots, histograms, and 3D plots, among others.

# How to Run the code
From the submitted document, there is the link to google colab. Inside google colab there are codes and text to make the notebook readable. The code in the notebook should be run in sequential order from the top to the bottom starting from libraries, to datasets, data pre-processing, model design, model training and validation, model testing and evaluation, and confusion matrix. In the notebook, the code is separated using appropriate headings to make the code easily understood. To reiterate, the code results from different model designs, because of different hyper-parameters due to the use of the random search method from the Keras-Tuner. However, the model do predict the exert value of the image even if the code is run how many times will results in high accuracy to show that indeed machine has learned. The following are the steps to run the notebook with provided link:

* Open your web browser and go to the Google Colab website (https://colab.research.google.com/).
* Click on the "GitHub" tab in the "Welcome to Colaboratory!" dialog box. This will allow you to run a notebook from a GitHub repository.
* In a separate tab, locate the notebook you want to run on GitHub. Copy the URL of the notebook's GitHub page.
* Go back to the Google Colab tab and paste the URL into the GitHub tab's search bar. Press Enter or click the magnifying glass icon to search for the notebook.
* The search results will display the available notebooks from the GitHub repository. Click on the notebook you want to run.
* The notebook will open in Google Colab. You can now view and edit the code if needed.
* To execute the notebook and run its code cells, click on the "Runtime" menu at the top of the Colab interface, then select "Run all" or use the "Run" button next to each code cell to execute them individually.
* As the notebook runs, you will see the output generated by each code cell displayed below it.

