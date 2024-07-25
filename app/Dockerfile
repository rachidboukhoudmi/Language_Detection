FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app.py /app/app.py
COPY ./model.pkl /app/model.pkl
COPY ./vectorizer.pkl /app/vectorizer.pkl

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
