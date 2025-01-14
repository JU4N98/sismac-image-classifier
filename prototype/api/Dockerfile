FROM python:3.6.10-slim-buster
WORKDIR /api
COPY . ./

RUN apt-get update \
    && pip install --upgrade pip \
    && pip install --timeout 3000 -r requirements.txt

EXPOSE 5000

RUN ls
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]
