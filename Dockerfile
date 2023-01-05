# Dockerfile, Image, Container

From python:3.10.9

EXPOSE 8501

ADD app.py .

ADD ./model ./model

ADD ./src ./src

ADD main.py .

ADD ./data ./data

COPY requirements.txt .

COPY builder.py .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install wandb

RUN cp builder.py /usr/local/lib/python3.10/site-packages/google/protobuf/internal/

ENV PYTHONPATH = "."

CMD python -m streamlit run ./app.py --server.fileWatcherType none