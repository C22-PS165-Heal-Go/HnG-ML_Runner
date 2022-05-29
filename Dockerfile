FROM python:3.10-buster

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY  . .

CMD ["python", "main.py"]