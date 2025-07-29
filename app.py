from flask import Flask, request, render_template
import pandas as pd
import joblib
import time
from prometheus_client import Counter, Histogram, Gauge
from prometheus_flask_exporter import PrometheusMetrics
from utils import extract_datetime_features, get_item_descriptions
import psutil
import numpy as np
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares

# Метрики
REQUEST_COUNT = Counter("requests_total", "Total number of requests")
REQUEST_LATENCY = Histogram("custom_request_latency_seconds_my", "Latency of requests")
MEMORY_USAGE = Gauge("memory_usage_mb", "Memory usage")
CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage")

# Загрузка разреженной матрицы взаимодействий
# Преобразование в формат implicit (транспонируется: items x users)
train = load_npz("data/train_matrix.npz").T.tocsr()
# train = train.T.tocsr()

# Маппинги пользователей и товаров
user_mapping = np.load("data/train_matrix_user_mapping.npy", allow_pickle=True).item()
item_mapping = np.load("data/train_matrix_item_mapping.npy", allow_pickle=True).item()

# Загрузка модели и данных
model = joblib.load("AlternatingLeastSquares.pkl")
descriptions = pd.read_csv("data/copy_properties_full_long.csv")


app = Flask(__name__)
metrics = PrometheusMetrics(app, defaults_prefix="flask")
metrics.register_default(
    metrics=REQUEST_LATENCY,
    endpoints=["/metrics"]
)

# не хочется ради красоты постоянно гонять данные туда сюда, поэтому пусть лежит здесь
def get_recommendations(user_id, n_recommendations=3):
    top_item = [461686, 119736, 213834]
    # , f"Пользователь {user_id} не найден в данных."
    if user_id not in user_mapping:
        return top_item
    
    user_idx = user_mapping[user_id]

    # , f"У пользователя {user_id} нет взаимодействий в обучающих данных."
    if train[user_idx].nnz == 0:
        return top_item
    
    recommended_item_indices, scores = model.recommend(user_idx, train[user_idx], N=n_recommendations)

    reverse_item_mapping = {idx: iid for iid, idx in item_mapping.items()}
    recommended_items = [reverse_item_mapping[idx] for idx in recommended_item_indices]
    
    return recommended_items


@app.route("/", methods=["GET", "POST"])
def index():
    REQUEST_COUNT.inc()

    # Обновим метрики ресурсов
    MEMORY_USAGE.set(psutil.virtual_memory().used / 1024**2)
    CPU_USAGE.set(psutil.cpu_percent())

    start_time = time.time()
    if request.method == "POST":
        user_input = request.form.get("user_id")
        if not user_input or not user_input.isdigit():
            REQUEST_LATENCY.observe(time.time() - start_time)
            return render_template("index.html", message="Пожалуйста, введите правильное значение.")
        if int(user_input) == 0:
            REQUEST_LATENCY.observe(time.time() - start_time)
            return render_template("index.html", message="Пожалуйста, введите правильное значение.")
        else:
            user_id = int(user_input)

            # Предсказание
            top_items = get_recommendations(user_id)
            descriptions_text = get_item_descriptions(top_items, descriptions)
            message = f'Рекомендации для пользователя <strong>{user_id}</strong>:<br><br>1) {top_items[0]}<br>2) {top_items[1]}<br>3) {top_items[2]}'

            REQUEST_LATENCY.observe(time.time() - start_time)
            return render_template("index.html", message=message, descriptions=descriptions_text)

    REQUEST_LATENCY.observe(time.time() - start_time)
    return render_template("index.html", message="Айди больше 0, целое положительное число")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
