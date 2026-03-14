FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model/ ./model/
COPY src/ ./src/

EXPOSE $PORT

CMD ["sh", "-c", "python src/main.py"]
