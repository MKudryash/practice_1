import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# 1. Загружаем предобученную модель ПРИ ЗАПУСКЕ СЕРВИСА
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

try:
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME)
    print("✅ Модель загружена успешно!")
except Exception as e:
    print(f"❌ Не удалось загрузить модель: {e}")
    exit(1)


# 2. Описываем структуру входящего JSON-запроса
class TextRequest(BaseModel):
    text: str  # Обязательное поле с текстом


# 3. Создаем экземпляр приложения FastAPI
app = FastAPI(
    title="ML Inference Service",
    description="Сервис для классификации тональности текста",
    version="1.0"
)


# 4. Эндпоинт для проверки здоровья сервиса (healthcheck)
@app.get("/health")
async def health_check():
    """
    Возвращает статус сервиса. Используется Docker для проверки, жив ли контейнер.
    """
    return {"status": "ok", "message": "Service is running"}


# 5. Эндпоинт для предсказания (работает через POST)
@app.post("/predict")
async def predict(request: TextRequest):
    """
    Принимает JSON вида {"text": "какой-то текст"}.
    Возвращает предсказанную метку (POSITIVE/NEGATIVE) и уверенность модели.
    """
    input_text = request.text

    # Проверка на пустую строку
    if not input_text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    # Делаем предсказание с помощью transformers pipeline
    result = classifier(input_text)[0]
    # result выглядит так: {'label': 'POSITIVE', 'score': 0.9998}

    return {
        "label": result["label"],       # "POSITIVE" или "NEGATIVE"
        "score": float(result["score"])  # уверенность модели (0..1)
    }


# 6. Точка входа для локального запуска (для отладки без Docker)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)