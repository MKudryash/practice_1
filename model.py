from transformers import pipeline

# Скачиваем и кешируем модель локально (~260 МБ).
# При первом запуске модель будет загружена из Hugging Face Hub
# и сохранена в кеш (~/.cache/huggingface), поэтому повторные
# запуски (в том числе внутри Docker, если кеш сохранён в образе)
# не будут скачивать её заново.
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

if __name__ == "__main__":
    print(f"⏳ Загружаю модель {MODEL_NAME}...")
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME)
    print("✅ Модель успешно загружена и закеширована!")

    # Быстрая проверка, что всё работает
    result = classifier("This is a great movie!")
    print("Тестовое предсказание:", result)