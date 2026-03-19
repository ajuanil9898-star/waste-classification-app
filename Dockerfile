FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY app ./app
COPY models ./models
COPY mlartifacts /mlartifacts

EXPOSE 10000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "10000"]