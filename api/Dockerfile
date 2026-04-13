FROM python:3.12-slim

WORKDIR /code
COPY ./api/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src /code/src

ENV PYTHONPATH=/code/src

COPY ./api/app /code/app

# Start the FastAPI application using the production-ready CLI
CMD ["fastapi", "run", "app/main.py", "--port", "80"]