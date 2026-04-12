FROM apache/airflow:3.1.8
USER root

RUN rm -rf /var/lib/apt/lists/* \
  && apt-get clean \
  && apt-get update -o Acquire::Retries=3 -o Acquire::http::Pipeline-Depth=0 \
  && apt-get install -y --no-install-recommends \
         vim \
         libgl1-mesa-glx \
         libglib2.0-0 \
         libsm6 \
         libxext6 \
         libxrender-dev \
         libxcb1 \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements and install them in the Airflow container
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Set the safe directory for git to avoid issues with Airflow's DVC integration
RUN git config --global --add safe.directory '*'