FROM python:3.6
COPY ./app/requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install gunicorn[gevent]
COPY ./app /app
CMD ["./run.sh"]