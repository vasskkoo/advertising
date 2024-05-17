# Імпортуємо необхідні бібліотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Приклад моделі
import joblib

def train_model(data_path, target_column, model_path):
    # Зчитуємо дані
    data = pd.read_csv(data_path)

    # Розділяємо дані на ознаки (X) та цільову змінну (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Розділяємо дані на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ініціалізуємо та навчаємо модель
    model = LinearRegression()  # Приклад моделі (в даному випадку - лінійна регресія)
    model.fit(X_train, y_train)  # Навчаємо модель на тренувальних даних

    # Зберігаємо навчену модель
    joblib.dump(model, model_path)
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    # Шлях до даних, назва цільової змінної та шлях для збереження моделі
    data_path = "/Users/mac/Desktop/rgr/data/new_adv.csv"
    target_column = "Sales"
    model_path = "/Users/mac/Desktop/rgr/models/regression_model.pkl"

    # Викликаємо функцію для тренування моделі
    train_model(data_path, target_column, model_path)