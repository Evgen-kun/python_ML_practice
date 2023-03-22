# Пример на языке Python для проведения предобработки данных и
# использования классических алгоритмов машинного обучения

# Импортируем необходимые библиотеки:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constants
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def feature_analysis(data):
    pd.set_option('display.max_columns', None)
    print("Вывести первичный анализ признаков?",
          " * 1 - Да",
          " * 2 - Нет", sep='\n')
    ans = int(input())
    if ans == 2:
        return
    # Первичный анализ признаков
    print(data.describe())
    print(data.describe())


def visual_feature_analysis(data):
    print("Вывести визуальный анализ признаков?",
          " * 1 - Да",
          " * 2 - Нет", sep='\n')
    ans = int(input())
    if ans == 2:
        return
    while True:
        print("Выберите для первичного визуального анализа:",
              " * 0 - EXIT",
              " * 1 - Age",
              " * 2 - fnlwgt",
              " * 3 - Education_Num",
              " * 4 - Capital_Gain",
              " * 5 - Capital_Loss",
              " * 6 - Hours_per_week",
              " * 7 - Target",
              " * 8 - Workclass",
              " * 9 - Education",
              " * 10 - Martial_Status",
              " * 11 - Occupation",
              " * 12 - Relationship",
              " * 13 - Race",
              " * 14 - Sex",
              " * 15 - Country",
              sep='\n')
        ans = int(input())
        if (ans > 15) or (ans < 1):
            break
        plt.hist(data[constants.ALL_FEATURES_COLLECTION.get(ans)])
        plt.show()


def visual_correlation_matrix(data):
    print("Вывести Корелляционную матрицу?",
          " * 1 - Да",
          " * 2 - Нет", sep='\n')
    ans = int(input())
    if ans == 2:
        return
    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()


# Используем метрику accuracy_score для вычисления точности, которая показывает долю правильных прогнозов.
def logistic_regression(data):
    x_train, x_test, y_train, y_test = data
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print("Logistic Regression Accuracy: ", accuracy_lr)


def decision_tree(data):
    x_train, x_test, y_train, y_test = data
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print("Decision Tree Accuracy: ", accuracy_dt)


def support_vector_machines(data):
    x_train, x_test, y_train, y_test = data
    svm = SVC()
    svm.fit(x_train, y_train)
    y_pred_svm = svm.predict(x_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("SVM Accuracy: ", accuracy_svm)


def k_nearest_neighbors(data):
    x_train, x_test, y_train, y_test = data
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("k-NN Accuracy: ", accuracy_knn)


def all_methods(data):
    logistic_regression(data)
    decision_tree(data)
    support_vector_machines(data)
    k_nearest_neighbors(data)


switcher = {
    1: logistic_regression,
    2: decision_tree,
    3: support_vector_machines,
    4: k_nearest_neighbors,
    0: all_methods
}


def encode_categorical_features(data):
    for feature in constants.CATEGORICAL_FEATURES:
        data[feature] = pd.Categorical(data[feature])
        data[feature] = data[feature].cat.codes


def load_from_csv():
    # Загрузим данные и разделим их на обучающую и тестовую выборки:
    adult_train = pd.read_csv('lab/11/adult_train.csv', sep=';')
    adult_test = pd.read_csv('lab/11/adult_test.csv', sep=';', skiprows=[1])
    # adult_modified = adult_train.set_index('fnlwgt')
    encode_categorical_features(adult_train)
    encode_categorical_features(adult_test)
    print(adult_train.head())
    print(adult_train.head())
    feature_analysis(adult_train)
    # Первичный визуальный анализ признаков
    visual_feature_analysis(adult_train)
    # Закономерности, особенности данных
    visual_correlation_matrix(adult_train)
    return [adult_train, adult_test]


def ml(data_from_csv):
    x_data, y_data = data_from_csv
    x = x_data.drop('Target', axis=1)
    y = x_data['Target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

    switcher.get(int(input()))(data)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    encode_data = load_from_csv()
    while True:
        ml(encode_data)
        print("Продолжить?", " * 1 - Да", " * Other number - Нет", sep='\n')
        if int(input()) != 1:
            break
