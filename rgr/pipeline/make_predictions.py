# Імпортуємо необхідні бібліотеки
import pandas as pd
import joblib

def make_predictions(new_data_path, model_path, output_path):
    # Зчитуємо нові дані
    new_data = pd.read_csv(new_data_path)

    # Завантажуємо навчену модель
    model = joblib.load(model_path)

    # Видаляємо ознаку "Sales" з нових даних
    new_data = new_data.drop(columns=['Sales'])

    # Здійснюємо передбачення
    predictions = model.predict(new_data)

    # Зберігаємо передбачення
    pd.DataFrame(predictions, columns=['Predictions']).to_csv(output_path, index=False)

if __name__ == "__main__":
    # Шляхи до нових даних, натренованої моделі та місця для збереження результатів передбачень
    new_data_path = "/Users/mac/Desktop/rgr/data/new_adv.csv"
    model_path = "/Users/mac/Desktop/rgr/models/regression_model.pkl"
    output_path = "/Users/mac/Desktop/rgr/results/predictions.csv"

    # Викликаємо функцію для здійснення передбачень
    make_predictions(new_data_path, model_path, output_path)