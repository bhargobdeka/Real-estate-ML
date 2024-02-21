FROM python:3.11-slim

COPY requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

EXPOSE 8080

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]