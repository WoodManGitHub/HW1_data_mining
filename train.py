import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
import os
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from rpy2.robjects import pandas2ri

pandas2ri.activate()


def load_and_prepare_dataset(path=''):
    df = pd.read_csv(path)
    # Convert ? to nan
    df['workclass'] = df['workclass'].replace('?', np.nan)
    df['occupation'] = df['occupation'].replace('?', np.nan)
    df['native-country'] = df['native-country'].replace('?', np.nan)
    # Remove NaN, duplicates data and not important data
    df.dropna(how='any', inplace=True)
    df = df.drop_duplicates()
    df = df.drop(['fnlwgt', 'education-num', 'marital-status', 'capital-gain', 'capital-loss'], axis=1)
    return df

def split_dataset(dataset, features, target, test_size=0.2, random_state=33):
    X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[target], test_size=test_size, random_state=random_state)
    dict_vectorizer = DictVectorizer(sparse=False)
    X_train = dict_vectorizer.fit_transform(X_train.to_dict(orient='records'))
    X_test = dict_vectorizer.transform(X_test.to_dict(orient='records'))
    return X_train, X_test, y_train, y_test, dict_vectorizer

# Draw confusion diagram
def draw_confusion(output='', real={}, pred={}, name=''):
    class_names = ['<=50K', '>50K']
    confusion_matrix_diagram = confusion_matrix(real, pred)
    confusion_matrix_diagram = pd.DataFrame(confusion_matrix_diagram, columns=class_names, index=class_names)
    confusion_matrix_plot = sns.heatmap(confusion_matrix_diagram, annot=True, fmt='d', cmap='Blues')
    fig = confusion_matrix_plot.get_figure()
    fig.savefig(f'./output/{output}/{name}_confusion.png')
    plt.clf()

# Draw tree diagram
def draw_tree(model, output, dict_vectorizer={}):
    class_names = ['<=50K', '>50K']
    tree = export_graphviz(model,
                        out_file=None,
                        feature_names=dict_vectorizer.feature_names_,
                        class_names=class_names,
                        rounded=True,
                        filled=True,
                        impurity=True)
    graph_from_dot_data(tree).write_png(f'./output/{output}/tree.png')

# Write metrics to csv
def save_metrics_to_csv(model_name, train_pred, test_pred):
    metrics = {
        'Model': model_name,
        'Train Accuracy': accuracy_score(y_train, train_pred),
        'Train Precision': precision_score(y_train, train_pred, pos_label='<=50K'),
        'Train Recall': recall_score(y_train, train_pred, pos_label='<=50K'),
        'Train F1 Score': f1_score(y_train, train_pred, pos_label='<=50K'),
        'Test Accuracy': accuracy_score(y_test, test_pred),
        'Test Precision': precision_score(y_test, test_pred, pos_label='<=50K'),
        'Test Recall': recall_score(y_test, test_pred, pos_label='<=50K'),
        'Test F1 Score': f1_score(y_test, test_pred, pos_label='<=50K')
    }

    df = pd.DataFrame([metrics])

    # Check if the CSV file already exists
    if not os.path.isfile('./output/model_metrics.csv'):
        df.to_csv('./output/model_metrics.csv', index=False, header=True)
    else:  # If the file already exists, append the data
        df.to_csv('./output/model_metrics.csv', mode='a', header=False, index=False)

def fit_and_score(model, model_name, X_train, X_test, y_train, y_test, dict_vectorizer):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, train_pred)
    precision = precision_score(y_train, train_pred, pos_label='<=50K')
    recall = recall_score(y_train, train_pred, pos_label='<=50K')
    f1 = f1_score(y_train, train_pred, pos_label='<=50K')
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, pos_label='<=50K')
    test_recall = recall_score(y_test, test_pred, pos_label='<=50K')
    test_f1 = f1_score(y_test, test_pred, pos_label='<=50K')
    print('----------------')
    print(f'Model: {model_name}')
    print(f'Train:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1: {f1}\n')
    print(f'Test:\nAccuracy: {test_accuracy}\nPrecision: {test_precision}\nRecall: {test_recall}\nf1: {test_f1}\n')
    save_metrics_to_csv(model_name, train_pred, test_pred)
    draw_confusion(model_name, y_train, train_pred, 'train')
    draw_confusion(model_name, y_test, test_pred, 'test')
    draw_tree(model, model_name, dict_vectorizer)

# Build C5.0 Model with R language
def train_r_c5_model(X_train, y_train, output_directory):
    robjects.r('library(C50)')  # C5.0套件
    X_train_df = pd.DataFrame(X_train, columns=dict_vectorizer.feature_names_)
    y_train_r = robjects.StrVector(y_train)  # 將y_train轉換為R字符向量
    robjects.r.assign("X_train", X_train_df)  # 將X_train_df傳遞給R
    robjects.r.assign("y_train", y_train_r)  # 將y_train傳遞給R
    robjects.r('model <- C5.0(x = X_train, y = factor(y_train))')  # 訓練C5.0模型
    text = str(robjects.r('summary(model)'))  # 生成結果圖表
    text = text.split('Decision tree:')
    text = text[1].split('Evaluation on training data')[0][2:]
    with open('./output/C5.0/tree.txt', 'w') as file:
        file.write(text)
    model = robjects.r('model')  # 將R中的模型返回給Python

    return model

# Evaluate C5.0 Model with R language
def evaluate_r_c5_model(model, X_train, y_train, X_test, y_test, output, dict_vectorizer):
    # Convert X_train to a data frame
    X_train_df = pd.DataFrame(X_train, columns=dict_vectorizer.feature_names_)
    robjects.r.assign("X_train", X_train_df)

    # Use data frame for prediction on training data
    robjects.r('train_pred <- predict(model, X_train, type="class")')
    train_pred_c5 = robjects.r('train_pred')
    train_pred_c5 = [x for x in train_pred_c5]
    train_pred_c5 = [">50K" if p == 2 else "<=50K" for p in train_pred_c5]  # 將預測結果重新映射為與真實標籤相同的類型

    # Convert X_test to a data frame
    X_test_df = pd.DataFrame(X_test, columns=dict_vectorizer.feature_names_)
    robjects.r.assign("X_test", X_test_df)

    # Use data frame for prediction on test data
    robjects.r('test_pred <- predict(model, X_test, type="class")')
    test_pred_c5 = robjects.r('test_pred')
    test_pred_c5 = [x for x in test_pred_c5]
    test_pred_c5 = [">50K" if p == 2 else "<=50K" for p in test_pred_c5]  # 將預測結果重新映射為與真實標籤相同的類型

    # Calculate accuracy for training data
    accuracy_c5 = accuracy_score(y_train, train_pred_c5)
    precision_c5 = precision_score(y_train, train_pred_c5, pos_label='<=50K')
    recall_c5 = recall_score(y_train, train_pred_c5, pos_label='<=50K')
    f1_c5 = f1_score(y_train, train_pred_c5, pos_label='<=50K')

    # Calculate accuracy for test data
    test_accuracy_c5 = accuracy_score(y_test, test_pred_c5)
    test_precision_c5 = precision_score(y_test, test_pred_c5, pos_label='<=50K')
    test_recall_c5 = recall_score(y_test, test_pred_c5, pos_label='<=50K')
    test_f1_c5 = f1_score(y_test, test_pred_c5, pos_label='<=50K')

    print('----------------')
    print('Model: C5.0 (R)')
    print(f'Train:\nAccuracy: {accuracy_c5}\nPrecision: {precision_c5}\nRecall: {recall_c5}\nf1: {f1_c5}\n')
    print(f'Test:\nAccuracy: {test_accuracy_c5}\nPrecision: {test_precision_c5}\nRecall: {test_recall_c5}\nf1: {test_f1_c5}\n')

    save_metrics_to_csv('C5.0', train_pred_c5, test_pred_c5)
    draw_confusion('C5.0', y_train, train_pred_c5, 'train')
    draw_confusion('C5.0', y_test, test_pred_c5, 'test')

# Train C4.5 Model with RWeka
def train_r_c45_model(X_train, y_train, output_directory):
    robjects.r('library(RWeka)')
    X_train_df = pd.DataFrame(X_train, columns=dict_vectorizer.feature_names_)
    y_train_r = robjects.StrVector(y_train)
    robjects.r.assign("X_train", X_train_df)
    robjects.r.assign("y_train", y_train_r)
    robjects.r('model <- J48(x = X_train, y = y_train)')
    text = str(robjects.r('summary(model)'))
    text = text.split('Decision tree:')
    text = text[1].split('Evaluation on training data')[0][2:]
    with open('./output/C4.5/tree.txt', 'w') as file:
        file.write(text)
    model = robjects.r('model')

    return model

# Evaluate C4.5 Model with RWeka
def evaluate_r_c45_model(model, X_train, y_train, X_test, y_test, output, dict_vectorizer):
    # Convert X_train to a data frame
    X_train_df = pd.DataFrame(X_train, columns=dict_vectorizer.feature_names_)
    robjects.r.assign("X_train", X_train_df)

    # Use data frame for prediction on training data
    robjects.r('train_pred <- predict(model, X_train)')
    train_pred_c45 = robjects.r('train_pred')
    train_pred_c45 = [x for x in train_pred_c45]

    # Convert X_test to a data frame
    X_test_df = pd.DataFrame(X_test, columns=dict_vectorizer.feature_names_)
    robjects.r.assign("X_test", X_test_df)

    # Use data frame for prediction on test data
    robjects.r('test_pred <- predict(model, X_test)')
    test_pred_c45 = robjects.r('test_pred')
    test_pred_c45 = [x for x in test_pred_c45]

    # Calculate accuracy for training data
    accuracy_c45 = accuracy_score(y_train, train_pred_c45)
    precision_c45 = precision_score(y_train, train_pred_c45, pos_label='<=50K')
    recall_c45 = recall_score(y_train, train_pred_c45, pos_label='<=50K')
    f1_c45 = f1_score(y_train, train_pred_c45, pos_label='<=50K')

    # Calculate accuracy for test data
    test_accuracy_c45 = accuracy_score(y_test, test_pred_c45)
    test_precision_c45 = precision_score(y_test, test_pred_c45, pos_label='<=50K')
    test_recall_c45 = recall_score(y_test, test_pred_c45, pos_label='<=50K')
    test_f1_c45 = f1_score(y_test, test_pred_c45, pos_label='<=50K')

    print('----------------')
    print('Model: C4.5 (RWeka)')
    print(f'Train:\nAccuracy: {accuracy_c45}\nPrecision: {precision_c45}\nRecall: {recall_c45}\nf1: {f1_c45}\n')
    print(f'Test:\nAccuracy: {test_accuracy_c45}\nPrecision: {test_precision_c45}\nRecall: {test_recall_c45}\nf1: {test_f1_c45}\n')

    draw_confusion('C4.5', y_train, train_pred_c45, 'train')
    draw_confusion('C4.5', y_test, test_pred_c45, 'test')

if __name__ == '__main__':
    # Load and prepare the dataset
    dataset = load_and_prepare_dataset('./datasets/adult.csv')

    features = ['age', 'workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country']
    target = 'income'

    # Split the dataset
    X_train, X_test, y_train, y_test, dict_vectorizer = split_dataset(dataset, features, target)

    depth = 3

    # Train ID3 Model
    ID3Tree = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    fit_and_score(ID3Tree, 'ID3', X_train, X_test, y_train, y_test, dict_vectorizer)

    # Train CART Model  
    CARTTree = DecisionTreeClassifier(max_depth=depth, criterion='gini')
    fit_and_score(CARTTree, 'CART', X_train, X_test, y_train, y_test, dict_vectorizer)

    # Train C4.5 Model
    C45_model = train_r_c45_model(X_train, y_train, output_directory='./output')
    evaluate_r_c45_model(C45_model, X_train, y_train, X_test, y_test, 'C4.5_R', dict_vectorizer)

    # Train C5.0 Model
    C5_model = train_r_c5_model(X_train, y_train, output_directory='./output')
    evaluate_r_c5_model(C5_model, X_train, y_train, X_test, y_test, 'C5.0_R', dict_vectorizer)
