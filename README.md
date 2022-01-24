**Positivity analysis for European restaurants reviews**


This model was developed by me and Marta Skrodzka-Paruzel as a final project for Data Science course at Software Development Academy. 

The [data file](https://www.kaggle.com/damienbeneschi/krakow-ta-restaurans-data-raw?select=TA_restaurants_curated.csv) was downloaded from [Kaggle](https://www.kaggle.com/) and it contains over 125k records with restaurants information for 31 cities in Europe. The model we developed and trained estimates the review sentiment: positive or non-positive. 

Technologies in use:
- Jupyter Notebook, PyCharm, GIT, GitHub
- numpy, pandas, regex
- matplotlib, plotly
- scikit-learn (LogisticRegression, BernoulliNB, KNeighborsClassifier)
- spacy, gensim (with glove-twitter-200 model)
- TensorFlow  + Keras (RNN networks with Embedding and LSTM layers)

To run the analysis you need to:
1. Download the files from this repository and install all the packages from requirements.txt file.

2. Download the [data file](https://www.kaggle.com/damienbeneschi/krakow-ta-restaurans-data-raw?select=TA_restaurants_curated.csv)
and save it in a folder named **data**.

  3. Run the **preprocessing.py** script to generate the preprocessed data file (you can use `python path_to_file\preprocessing.py` command in any command prompt).

Alternatively you can use the preprocessed data file from [here](https://drive.google.com/drive/folders/1EBKB7KR7cryLxCCIFB2vaTCioDwc5wH6?usp=sharing).

4. Run the **model_impl.py** script.


For more detailed project description in Polish see Projekt SDA.pptx file.
