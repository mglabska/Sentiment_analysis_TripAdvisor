import pandas as pd
import regex as re
import spacy
import string


class Preprocess:
    def __init__(self, data, column, ccolumn, filter_col):
        self.data = data
        self.column = column
        self.ccolumn = ccolumn
        self.filter_col = filter_col

    def preproc_cuisine_style(self):
        self.data[self.ccolumn] = self.data[self.ccolumn].fillna("missing values")
        self.data[self.ccolumn] = self.data[self.ccolumn].apply(lambda x: re.sub('\[', '', str(x)))
        self.data[self.ccolumn] = self.data[self.ccolumn].apply(lambda x: re.sub('\]', '', str(x)))
        self.data[self.ccolumn] = self.data[self.ccolumn].apply(lambda x: x.split(','))
        self.data = pd.concat([self.data, pd.get_dummies(self.data[self.ccolumn].apply(pd.Series), prefix='',
                                                         prefix_sep='').groupby(level=0, axis=1).sum()], axis=1)
        self.data.pop(self.ccolumn)

    def preproc_reviews(self):
        self.data[self.column] = self.data[self.column].apply(lambda x: re.findall('\[\[.*?\]', str(x)))
        self.data[self.column] = self.data[self.column].apply(lambda x: re.sub('[\\\[\]"]', '', str(x)))
        self.data[['temp1', 'temp2']] = self.data[self.column].str.split("', '", 1, expand=True)
        data_1 = self.data.drop([self.column, 'temp2'], axis=1).rename(columns={'temp1': self.column})
        data_2 = self.data.drop([self.column, 'temp1'], axis=1).rename(columns={'temp2': self.column})
        data_2 = data_2.drop(data_2.loc[data_2[self.filter_col] > 3.0].index)
        self.data = data_1.append(data_2).dropna()

    def get_data(self):
        self.preproc_reviews()
        self.preproc_cuisine_style()
        return self.data


def create_categories(filter_col):
    if filter_col <= 3.0:
        return 0
    elif filter_col >= 4.5:
        return 1
    else:
        return None


def clean_text(data_text):
    text = nlp_en_sm(data_text)
    string_2 = ''
    for token in text:
        string_2 = string_2 + ' ' + token.lemma_
    data_clean = string_2
    data_clean = data_clean.split()
    data_clean = [word for word in data_clean if word not in stop_words]
    data_clean = ' '.join(map(str, data_clean))
    data_clean = re.sub("'", "", data_clean)
    return data_clean.strip(string.punctuation)


if __name__ == '__main__':
    nlp_en_sm = spacy.load("en_core_web_sm")
    stop_words = nlp_en_sm.Defaults.stop_words
    prep_data = pd.read_csv('data/TA_restaurants_curated.csv', encoding='utf-8')[
        ['Reviews', 'Cuisine Style', 'City', 'Rating']
    ]
    prep_data = prep_data.drop(prep_data.loc[prep_data['Reviews'] == '[[], []]'].index).dropna()
    prep_data = prep_data.drop(prep_data.loc[prep_data['Rating'] == -1].index).dropna()
    pdata = Preprocess(prep_data, 'Reviews', 'Cuisine Style', 'Rating')
    final_data = pdata.get_data()
    final_data['Category'] = final_data['Rating'].apply(create_categories)
    final_data["Reviews_cleaned"] = final_data["Reviews"].apply(clean_text)
    final_data = final_data.dropna()
    final_data.to_csv('data/preprocessed.csv')
