import requests
import time
import statistics

# Адрес нашего сервера (после запуска внутри Docker он будет доступен по localhost)
URL = "http://localhost:8000/predict"

# Модель distilbert-base-uncased-finetuned-sst-2-english обучена на английском,
# поэтому тестовые фразы тоже на английском — на русском тексте
# качество классификации будет непредсказуемым.
test_phrases = [
    "This is absolutely terrible, I'm so upset",
    "It's okay, nothing special",
    "Amazing, thank you so much!"
]


def test_performance():
    latencies = []
    print("🚀 Запуск нагрузочного теста из 10 запросов...")

    for i in range(10):
        # Берем фразу по кругу (i % 3)
        text = test_phrases[i % len(test_phrases)]

        start_time = time.time()

        try:
            response = requests.post(URL, json={"text": text}, timeout=5)
            response.raise_for_status()  # выбросит исключение, если статус не 2xx

            # Если ответ пришел, считаем время
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # переводим в миллисекунды
            latencies.append(latency)

            result = response.json()
            label = result.get("label", "?")
            score = result.get("score", 0.0)

            # Выводим результат запроса
            print(
                f"Запрос {i+1}: текст='{text[:30]}...' -> "
                f"{label} ({score:.4f}) (задержка: {latency:.2f} мс)"
            )

        except requests.exceptions.RequestException as e:
            print(f"❌ Ошибка запроса: {e}")

    if latencies:
        print("\n📊 СТАТИСТИКА ЗАДЕРЖЕК:")
        print(f" Средняя: {statistics.mean(latencies):.2f} мс")
        print(f" Медиана: {statistics.median(latencies):.2f} мс")
        print(f" Минимальная: {min(latencies):.2f} мс")
        print(f" Максимальная: {max(latencies):.2f} мс")
        if len(latencies) > 1:
            print(f" Стандартное отклонение: {statistics.stdev(latencies):.2f} мс")


if __name__ == "__main__":
    test_performance()