FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-xla.2-1.py310

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY . /app

CMD ["python", "src/task.py", "--help"]
