from tqdm import tqdm
import json
import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from lexica.util import *


class Dataset():
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.shuffle()

    def shuffle(self):
        np.random.seed(1117)
        np.random.shuffle(self.X)
        np.random.seed(1117)
        np.random.shuffle(self.y)


# ['checkins', 'bio', 'photo', 'canonicalUrl', 'firstName', 'lastName', 'lists', 'homeCity', 'photos', 'type', 'contact', 'canonicalPath', 'gender', 'isAnonymous', 'tips', 'mayorships', 'friends', 'id', 'lenses']
def get_features(rpath, scaler=None):
    data = []
    y = []
    venue_types = [
        '-',
        'unknown',
        'College_University',
        'Event',
        'Food',
        'Nightlife_Spot',
        'Outdoors_Recreation',
        'Professional_Other_Places',
        'Residence',
        'Shop_Service',
        'Travel_Transport',
        'Arts_Entertainment'
    ]
    liwc_categories = [
        'WC', 'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS', 'Sixltr', 'Dic', 'function', 'pronoun', 'ppron', 'i',
        'we', 'you', 'shehe', 'they', 'ipron', 'article', 'prep', 'auxverb', 'adverb', 'conj', 'negate', 'verb', 'adj',
        'compare', 'interrog', 'number', 'quant', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social',
        'family',
        'friend', 'female', 'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'differ', 'percept',
        'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation', 'achieve', 'power',
        'reward', 'risk', 'focuspast', 'focuspresent', 'focusfuture', 'relativ', 'motion', 'space', 'time', 'work',
        'leisure', 'home', 'money', 'relig', 'death', 'informal', 'swear', 'netspeak', 'assent', 'nonflu', 'filler',
        'AllPunc', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth',
        'OtherP'
    ]

    sid = SentimentIntensityAnalyzer()
    lex = liwc.parse_liwc("2015")
    with open(rpath) as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            try:
                u = json.loads(line)
                tips = u['fsq']['tips']['tips content']

                if not len(tips):
                    tip_features = dict(
                        tip_len=[0],
                        sentiment_pos=[0],
                        sentiment_neg=[0],
                        sentiment_neu=[0],
                        sentiment_compound=[0],
                        venue_types=[0]
                    )
                    for cat in liwc_categories:
                        tip_features['liwc_' + cat] = [0]
                else:
                    tip_features = dict(
                        tip_len=[],
                        sentiment_pos=[],
                        sentiment_neg=[],
                        sentiment_neu=[],
                        sentiment_compound=[],
                        venue_types=[]
                    )
                    for cat in liwc_categories:
                        tip_features['liwc_' + cat] = []

                    for t in tips:
                        tip_features['tip_len'].append(len(t['text'].split(' ')))
                        ss = sid.polarity_scores(t['text'])
                        tip_features['sentiment_pos'].append(ss['pos'])
                        tip_features['sentiment_neg'].append(ss['neg'])
                        tip_features['sentiment_neu'].append(ss['neu'])
                        tip_features['sentiment_compound'].append(ss['compound'])
                        tip_features['venue_types'].append(t['category'])
                        liwc_features = extract(lex, t['text'])
                        for cat in liwc_categories:
                            if cat in liwc_features:
                                tip_features['liwc_' + cat].append(liwc_features[cat])
                            else:
                                tip_features['liwc_' + cat].append(0.)

                feature = dict(
                    # bio_len=len(u['fsq']['bio'].split(' ')),
                    # n_lists=u['fsq']['lists']['count'],
                    # n_checkins=u['fsq']['checkins']['count'],
                    # n_friends=u['fsq']['friends']['count'],
                    # n_tips=len(u['fsq']['tips']['tips content']),
                    avg_tip_len=np.average(tip_features['tip_len']),
                    avg_sentiment_compound=np.nanmean(tip_features['sentiment_compound']),
                    avg_sentiment_pos=np.average(tip_features['sentiment_pos']),
                    avg_sentiment_neg=np.average(tip_features['sentiment_neg']),
                    avg_sentiment_neu=np.average(tip_features['sentiment_neu']),
                    std_tip_len=np.std(tip_features['tip_len']),
                    std_sentiment_compound=np.std(tip_features['sentiment_compound']),
                    std_sentiment_pos=np.std(tip_features['sentiment_pos']),
                    std_sentiment_neg=np.std(tip_features['sentiment_neg']),
                    std_sentiment_neu=np.std(tip_features['sentiment_neu'])
                )

                for type in venue_types:
                    feature['venue_type_' + type] = 1. * tip_features['venue_types'].count(type) / len(
                        tip_features['venue_types'])
                for cat in liwc_categories:
                    feature['avg_liwc_' + cat] = np.average(tip_features['liwc_' + cat])
                    feature['std_liwc_' + cat] = np.std(tip_features['liwc_' + cat])

                data.append(feature)
                y.append(u['label'])

            except Exception as ex:
                print(ex)

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(data)
    if not scaler:
        scaler = preprocessing.MinMaxScaler().fit(X)
    X = scaler.transform(X)

    feature_names = vectorizer.feature_names_
    y = np.array(y)

    return X, y, feature_names, scaler


def print_feature_importance(importance, feature_names):
    feature = {}
    for i in range(len(feature_names)):
        feature[feature_names[i]] = importance[i]

    feature = sorted(feature.items(), key=lambda x: x[1], reverse=True)

    for i in range(len(feature)):
        if feature[i][1] > 0:
            print(feature[i][0], '->', feature[i][1])
        else:
            break


def grid_search_cv(X_train, y_train, estimator, params, cv=5, scoring='roc_auc'):
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(estimator, params, cv=cv, scoring=scoring, n_jobs=-1)
    # grid.fit(X_train, y_train, cat_features=cat_feature_index)
    grid.fit(X_train, y_train)
    print('The parameters of the best model are: ')
    print("best parameters:", grid.best_params_, "best score:", grid.best_score_)
    return grid.best_estimator_


def evaluation(y_true, y_pred):
    from sklearn import metrics
    print("ACC:", metrics.accuracy_score(y_true, y_pred))
    print("AUC:", metrics.roc_auc_score(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred, digits=4))


def XGBoost(X_train, y_train, X_test, y_test, feature_names):
    from xgboost.sklearn import XGBClassifier
    params = {
        # 'learning_rate': [0.1, 0.03, 0.02, 0.01]
        # 'n_estimators': range(50, 200, 10),
        # 'max_depth': range(3,11,1),
        # 'min_child_weight': range(1,20,1),
        # 'gamma': [i/10.0 for i in range(0,10)]
        # 'subsample': [i / 10.0 for i in range(1, 10)],
        # 'colsample_bytree': [i / 10.0 for i in range(1, 10)]
        # 'seed': [0,50]
    }

    print("------------- XGBoost ------------")
    estimator = XGBClassifier(
        learning_rate=0.01,
        n_estimators=130,
        max_depth=5,
        min_child_weight=13,
        gamma=0.7,
        subsample=0.6,
        colsample_bytree=0.4,
        # scale_pos_weight=1,
        objective='binary:logistic',
        seed=0
    )
    clf = grid_search_cv(X_train, y_train, estimator, params)
    print_feature_importance(clf.feature_importances_, feature_names)
    y_pred = clf.predict(X_test)
    evaluation(y_test, y_pred)


if __name__ == '__main__':
    X_train, y_train, features_names, scaler = get_features('/bdata/jiayunz/Foursquare/100w/train_1_sentence_emb.json')
    print(features_names)
    trainset = Dataset(X_train, y_train, features_names)
    X_test, y_test, features_names, _ = get_features('/bdata/jiayunz/Foursquare/100w/test_1_sentence_emb.json', scaler)
    testset = Dataset(X_test, y_test, features_names)
    XGBoost(X_train, y_train, X_test, y_test, trainset.feature_names)