FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="UFold-repro" \
      description="UFold RNA secondary structure prediction"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install OS-level dependencies (opencv needs libgl)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy project source code
COPY Network.py .
COPY ufold_train.py .
COPY ufold_train_rivals.py .
COPY ufold_test.py .
COPY ufold_predict.py .
COPY process_data_newdataset.py .
COPY test_env.py .
COPY ufold/ ufold/

# Create directories for data, models, and results
RUN mkdir -p data models results

# Default: show usage help
CMD ["python", "-c", "print('UFold container ready.\\n\\nUsage examples:\\n  Training:    python ufold_train.py\\n  Rivals:      python ufold_train_rivals.py --gpu 0 --data_dir /app/data\\n  Testing:     python ufold_test.py --test_files TS2\\n  Prediction:  python ufold_predict.py\\n  Env check:   python test_env.py')"]
