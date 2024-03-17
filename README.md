## Распознование кошек и собак по фото
# Постановка задачи
Стоит задача классификации кошек и собак. Использован [датасет](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data) из kaggle
# Модель и пайплайн
Данные состоят из картинок кошек и собак, пайплайн выглядит так: возьмем замороженные веса ResNet18 и с помощью Transfer Learning повысим точность нашей модели дообучив классификатор
# Структура репозитория
```
├── dataset                     -- датасет
  ├── ...
├── DatasetLoader               -- загрузка датасета в удобном формате
│ ├── __init__.py
│ ├── ImageLoader.py            -- класс для обработки датасета
│ └── LoadDataset.py            -- загрузка и разделение датасета на train и test
├── Training_and_Inference      -- папка с обучением модели
  ├── Inference.py              -- использование обученной модели, смотрим на accuracy
  ├── Train.py                  -- обучение нашей нейронки, используем resnet18
  └── utils.py                  -- служебные функции
├── poetry.lock
├── pyproject.toml 
├── README.md
├── commands.py                 -- точка входа нашей программы
└── Dockerfile
```
# Сборка и запуск
* Запуск производится на системе Линукс
  * С использованием Docker:
    ```
    docker build -t neural .
    docker run neural
    ```
  * Без использования Docker, должен быть заранее предустановленный Poetry
    ```angular2html
    poetry install
    poetry run python3 commands.py
    ```