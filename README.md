## **Прогнозирование осадков с помощью машинного обучения**

В данном проекте была проведена работа по предсказанию осадков на основе исторических данных, используя методы машинного обучения. Основной целью было создание модели, способной классифицировать наличие или отсутствие осадков, а также оценка ее производительности.
Для достижения этой цели использовались различные алгоритмы, такие как логистическая регрессия, XGBoost и метод опорных векторов (SVC). Процесс включал в себя сбор данных, их предварительную обработку, обучение моделей и оценку их эффективности.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

"""Импортируем dataset"""

# из библиотеки google.colab импортируем класс files
from google.colab import files

# создаем объект этого класса, применяем метод .upload()
uploaded = files.upload()

Теперь загрузим набор данных и напечатаем его первые пять строк.

df = pd.read_csv('Rainfall.csv')
df.head()

"""Теперь проверим размер набора данных."""

df.shape

"""Проверим какой столбец набора данных содержит какой тип данных."""

df.info()

"""Согласно информации выше, мы можем заметить, что нет нулевых значений."""

df.describe().T

"""# Очистка Данных

Данные, полученные из первоисточников, называются необработанными данными и требуют большой предварительной обработки, прежде чем мы сможем сделать из них какие-либо выводы или провести некоторое моделирование. Эти этапы предварительной обработки известны как очистка данных и включают в себя удаление выбросов, усчисление нулевого значения и устранение любых расхождений во входных данных.
"""

df.isnull().sum()

"""Таким образом, в столбце «winddirection», а также в столбце «windspeed» есть одно нулевое значение. Но что случилось с названием столбца направление ветра?"""

df.columns

"""Здесь мы можем заметить, что в именах столбцов есть ненужные пробелы, давайте удалим это."""

df.rename(str.strip,
          axis='columns',
          inplace=True)

df.columns

"""Теперь пришло время для имутации нулевого значения."""

for col in df.columns:

  # Checking if the column contains
  # any null values
  if df[col].isnull().sum() > 0:
    val = df[col].mean()
    df[col] = df[col].fillna(val)

df.isnull().sum().sum()

"""# Анализ Исследовательский Данных

EDA - это подход к анализу данных с использованием визуальных методов. Он используется для обнаружения тенденций и закономерностей или для проверки предположений с помощью статистических резюме и графических представлений. Здесь мы увидим, как проверить дисбаланс данных и перекос данных.
"""

plt.pie(df['rainfall'].value_counts().values,
        labels = df['rainfall'].value_counts().index,
        autopct='%1.1f%%')
plt.show()

df.groupby('rainfall').mean()

"""Здесь мы можем четко провести некоторые наблюдения:

*   максимальная температура относительно ниже в дни осадков
*   значение точки росы выше в дни осадков
*   влажность высокая в дни, когда ожидается дождь
*   очевидно, что там должны быть облака для осадков
*   солнечного света также меньше в дни дождей
*   скорость ветра выше в дни дождей

Наблюдения, которые мы взяли из вышеуказанного набора данных, очень похожи на то, что наблюдается и в реальной жизни.
"""

features = list(df.select_dtypes(include = np.number).columns)
features.remove('day')
print(features)

"""Проверим распределение непрерывных функций, приведенных в наборе данных."""

plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.distplot(df[col])
plt.tight_layout()
plt.show()

"""Нарисуем boxplots для непрерывной переменной, чтобы обнаружить отложения, присутствующие в данных."""

plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.boxplot(df[col])
plt.tight_layout()
plt.show()

"""В данных есть отходы, но, к сожалению, у нас не так много данных, поэтому мы не можем их удалить."""

df.replace({'yes':1, 'no':0}, inplace=True)

"""Иногда существуют высококоррелированные особенности, которые просто увеличивают размерность пространства функций и не полезны для производительности модели. Поэтому мы должны проверить, есть ли в этом наборе данных высококоррелированные функции или нет."""

plt.figure(figsize=(10,10))
sb.heatmap(df.corr() > 0.8,
           annot=True,
           cbar=False)
plt.show()

"""Теперь удалим сильно связанные функции «maxtemp» и «mintemp». Но почему не температура или точка росы? Это связано с тем, что температура и точка росы предоставляют разную информацию о погоде и атмосферных условиях."""

df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)

"""# Модельное обучение

Теперь разделим функции и целевые переменные и разделим их на обучающие и тестовые данные, используя которые мы выберем модель, которая лучше всего работает с данными проверки.
"""

features = df.drop(['day', 'rainfall'], axis=1)
target = df.rainfall

"""Как и обнаружили ранее, набор данных, который использовали, был сбалансирован, поэтому нам придется сбалансировать обучающие данные, прежде чем подавать их в модель."""

X_train, X_val, \
    Y_train, Y_val = train_test_split(features,
                                      target,
                                      test_size=0.2,
                                      stratify=target,
                                      random_state=2)

# As the data was highly imbalanced we will
# balance it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority',
                        random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)

"""Особенности набора данных были в разном масштабе, поэтому нормализация его перед обучением поможет нам быстрее получить оптимальные результаты наряду со стабильным обучением."""

# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

"""Теперь давайте обучим некоторые современные модели для классификации и обучим их на наших обучающих данных.


*   Логистическая Регрессия
*   Классификатор XGB
*   СВК




"""

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

for i in range(3):
  models[i].fit(X, Y)

  print(f'{models[i]} : ')

  train_preds = models[i].predict_proba(X)
  print('Training Accuracy : ', metrics.roc_auc_score(Y, train_preds[:,1]))

  val_preds = models[i].predict_proba(X_val)
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, val_preds[:,1]))
  print()

"""# Оценка модели

Из приведенной выше точности мы можем сказать, что логистическая регрессия и классификатор вспомогательных векторов являются удовлетворительными, поскольку разрыв между обучением и точностью валидации низок. Давайте также нарисуем матрицу путаницы для данных проверки с использованием модели SVC.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

ConfusionMatrixDisplay.from_estimator(models[2], X_val, Y_val)
plt.show()

# This code is modified by Susobhan Akhuli

"""Давайте также назовем отчет о классификации для данных проверки с использованием модели SVC."""

print(metrics.classification_report(Y_val,
                                    models[2].predict(X_val)))

"""# Вывод

В результате проведенного анализа и обучения моделей удалось создать эффективную систему предсказания осадков. Сравнение различных алгоритмов показало, что каждая модель имеет свои сильные и слабые стороны, что подчеркивает важность выбора подходящего метода в зависимости от специфики задачи. Полученные результаты могут быть использованы для дальнейшего улучшения моделей и более точного предсказания осадков в будущем. Проект продемонстрировал значимость предварительной обработки данных и оценки моделей в процессе машинного обучения.
"""
