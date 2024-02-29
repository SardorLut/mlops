# Задача
Стоит задача детекции пневмонии по картинке. Использован [датасет](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) при этом все данные удобно разделены на train + test + val.
# Модель
Будет использована модель pretrained модель ResNet на которой мы дообучимся, возьмем ее из Pytorch. Мы выбираем ResNet, так как она показывает хороший [перфоманс](https://www.sciencedirect.com/science/article/pii/S0169743922000454).
