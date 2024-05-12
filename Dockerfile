FROM python:3-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt || true

COPY . .

ENTRYPOINT [ "python3", "app.py" ]
