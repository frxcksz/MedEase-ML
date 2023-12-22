## MedEase - Machine Learning
Here are the machine learning codes, models, and dataset for the MedEase project.

------------

### Overview
We build a Named-Entity Recognition (NER) model using spaCy to classify the text obtained from the Optical Character Recognition (OCR) model using keras-ocr, pre-processing the data using TensorFlow, labeling the data using BIO Tagging, and post-processing the data using Pandas and OpenCV. Also, providing JSON output for further processing or use.

------------

### Build With
- [spaCy](https://spacy.io/models#quickstart)
- [keras-ocr](https://keras-ocr.readthedocs.io/en/latest/)
- [TensorFlow](https://www.tensorflow.org/)

------------

### Other Libraries
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

------------

### Dataset
We collected images and medicine informations from data scattered across the internet and health-related apps.

------------

#### Run Project in Local
- [ ] Clone this Github [repository](https://github.com/frxcksz/MedEase-ML.git)
- [ ] Go to the project directory `cd MedEase-ML`
- [ ] Install the requirements `pip install MedEase-ML/requirements.txt`
- [ ] Put the image into the `MedEase-ML/tmp` directory
- [ ] Run python predictions.py
- [ ] The result of the image with bounding box will be displayed and you can see the JSON result in the `MedEase/result` directory
- [ ] Done!

------------

### Prerequisites
- [Python 3.11.0](https://www.python.org/downloads/release/python-3110/)
