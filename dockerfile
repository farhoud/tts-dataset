FROM tcb-base

ADD ./transcriber.py /app/transcriber.py

WORKDIR /app

VOLUME /app/data

CMD python3 transcriber.py



