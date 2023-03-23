# Пример на языке Python для проведения предобработки данных и
# использования классических алгоритмов машинного обучения

# Импортируем необходимые библиотеки:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constants
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


adult_train = pd
adult_test = pd
adult_train_modified = pd
adult_test_modified = pd


def feature_analysis():
    pd.set_option('display.max_columns', None)
    # print("Вывести первичный анализ признаков?",
    #       " * 1 - Да",
    #       " * 2 - Нет", sep='\n')
    # ans = int(input())
    # if ans == 2:
    #     return
    # Первичный анализ признаков
    print(adult_train_modified.describe())
    print(adult_train_modified.describe())


# Первичный визуальный анализ признаков
def visual_feature_analysis():
    global adult_train_modified, adult_test_modified
    while True:
        print("Выберите для первичного визуального анализа:")
        for col in adult_train_modified.columns:
            key = list(constants.ALL_FEATURES_COLLECTION.keys())[list(constants.ALL_FEATURES_COLLECTION.values()).index(col)]
            print(f" * {key} - {col}")
        ans = int(input())
        if (ans > 15) or (ans < 1):
            break
        plt.hist(adult_train_modified[constants.ALL_FEATURES_COLLECTION.get(ans)])
        plt.show()


# Закономерности, особенности данных
def visual_correlation_matrix():
    f = plt.figure(figsize=(19, 15))
    plt.matshow(adult_train_modified.corr(), fignum=f.number)
    plt.xticks(range(adult_train_modified.select_dtypes(['number']).shape[1]), adult_train_modified.select_dtypes(['number']).columns,
               fontsize=14, rotation=45)
    plt.yticks(range(adult_train_modified.select_dtypes(['number']).shape[1]), adult_train_modified.select_dtypes(['number']).columns,
               fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


def cross_validation(data):
    model, x, y, method = data
    print(f'Провести кросс-валидацию для метода "{method}"?',
          " * 1 - Да",
          " * 2 - Нет", sep='\n')
    ans = int(input())
    if ans == 2:
        return
    scores = cross_val_score(model, x, y, cv=5)  # 5-кратная кросс-валидация
    print('Оценки качества: ', scores)
    print('Среднее: ', scores.mean())
    print('Стандартное отклонение: ', scores.std())


# Используем метрику accuracy_score для вычисления точности, которая показывает долю правильных прогнозов.
def logistic_regression(data):
    x_train, x_test, y_train, y_test = data
    lr = LogisticRegression()
    cross_validation([lr, x_train, y_train, "логистическая регрессия"])
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print("Logistic Regression Accuracy: ", accuracy_lr)


def decision_tree(data):
    x_train, x_test, y_train, y_test = data
    dt = DecisionTreeClassifier()
    cross_validation([dt, x_train, y_train, "дерево решений"])
    dt.fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print("Decision Tree Accuracy: ", accuracy_dt)


def support_vector_machines(data):
    x_train, x_test, y_train, y_test = data
    svm = SVC()
    cross_validation([svm, x_train, y_train, "опорных векторов"])
    svm.fit(x_train, y_train)
    y_pred_svm = svm.predict(x_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("SVM Accuracy: ", accuracy_svm)


def k_nearest_neighbors(data):
    x_train, x_test, y_train, y_test = data
    knn = KNeighborsClassifier()
    cross_validation([knn, x_train, y_train, "k-ближайших соседей"])
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("k-NN Accuracy: ", accuracy_knn)


def all_methods(data):
    logistic_regression(data)
    decision_tree(data)
    support_vector_machines(data)
    k_nearest_neighbors(data)


def ml(data_from_csv):
    x_data, y_data = data_from_csv
    # x = x_data.drop('Target', axis=1)
    # y = x_data['Target']
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = x_data.drop('Target', axis=1)  # признаки для обучения модели
    y_train = x_data['Target']  # метки для обучения модели
    x_test = y_data.drop('Target', axis=1)  # признаки для тестирования модел
    y_test = y_data['Target']  # метки для тестирования модел

    # Предобработка данных. В этом примере мы используем стандартизацию данных:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    data = [x_train, x_test, y_train, y_test]

    # Теперь мы готовы использовать различные алгоритмы машинного обучения.
    # В этой программе мы используем логистическую регрессию,
    # дерево решений, метод опорных векторов и метод k-ближайших соседей:

    print("Выберите метод:",
          " * 1 - Логистическая регрессия",
          " * 2 - Дерево решений",
          " * 3 - Метод опорных векторов",
          " * 4 - k-ближайших соседей",
          " * 0 - Все сразу", sep='\n')

    method_switcher.get(int(input()))(data)


def encode_categorical_features(data):
    for feature in constants.CATEGORICAL_FEATURES:
        data[feature] = pd.Categorical(data[feature])
        data[feature] = data[feature].cat.codes


def feature_filter():
    print("Введите через запятую номера признаков, которые нужно убрать")
    global adult_train_modified, adult_test_modified
    print("Текущие признаки:")
    print(" * 0 - Отчистить фильтр")
    for col in adult_train_modified.columns:
        key = list(constants.ALL_FEATURES_COLLECTION.keys())[list(constants.ALL_FEATURES_COLLECTION.values()).index(col)]
        print(f" * {key} - {col}")
    print(" * 'q' - Выйти")
    ans = str(input()).split(',')
    if ans[0] == 'q':
        return
    if int(ans[0]) == 0:
        feature_filter_reset()
        return
    for feature_num in ans:
        adult_train_modified = adult_train_modified.drop(constants.ALL_FEATURES_COLLECTION.get(int(feature_num)), axis=1)
        adult_test_modified = adult_test_modified.drop(constants.ALL_FEATURES_COLLECTION.get(int(feature_num)), axis=1)


def feature_filter_reset():
    global adult_train, adult_test, adult_train_modified, adult_test_modified
    adult_train_modified = adult_train
    adult_test_modified = adult_test


method_switcher = {
    1: logistic_regression,
    2: decision_tree,
    3: support_vector_machines,
    4: k_nearest_neighbors,
    0: all_methods
}


analysis_switcher = {
    1: feature_analysis,
    2: visual_feature_analysis,
    3: visual_correlation_matrix,
    4: feature_filter
}


def load_from_csv():
    # Загрузим данные и разделим их на обучающую и тестовую выборки:
    global adult_train, adult_train_modified, adult_test, adult_test_modified
    adult_train = pd.read_csv('lab/11/adult_train.csv', sep=';')
    adult_train_modified = pd.read_csv('lab/11/adult_train.csv', sep=';')
    adult_test = pd.read_csv('lab/11/adult_test.csv', sep=';', skiprows=[1])
    adult_test_modified = pd.read_csv('lab/11/adult_test.csv', sep=';', skiprows=[1])
    # adult_modified = adult_train.set_index('fnlwgt')
    encode_categorical_features(adult_train)
    encode_categorical_features(adult_test)
    encode_categorical_features(adult_train_modified)
    encode_categorical_features(adult_test_modified)
    print(adult_train.head())
    print(adult_train.head())
    return [adult_train_modified, adult_test_modified]


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}!')  # Press Ctrl+F8 to toggle the breakpoint.


def main_menu(data):
    print_hi('PyCharm')
    while True:
        print("Выберите действие: ",
              " * 1 - Первичный анализ признаков",
              " * 2 - Первичный визуальный анализ признаков",
              " * 3 - Вывести корелляционную матрицу",
              " * 4 - Провести подбор параметров",
              " * 5 - Выбрать алгоритм машинного обучения",
              " * 6 - Выйти",
              sep='\n')
        ans = int(input())
        if ans < 1 or ans > 5:
            print("Завершение программы... ОК")
            return
        if ans == 5:
            ml(data)
            continue
        analysis_switcher.get(ans)()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_menu(load_from_csv())
