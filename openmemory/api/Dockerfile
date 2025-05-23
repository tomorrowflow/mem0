FROM python:3.12-slim

WORKDIR /usr/src/openmemory

# Copy requirements and install dependencies (including external mem0ai)
COPY ./openmemory/api/requirements.txt .
RUN pip install -r requirements.txt

# Find where mem0 was installed and overwrite with our local version
RUN python -c "import mem0; import os; print('mem0 installed at:', os.path.dirname(mem0.__file__))" > /tmp/mem0_location.txt

# Copy our local mem0 directory
COPY ./mem0 ./local_mem0

# Overwrite the installed mem0 package with our local version
RUN MEM0_PATH=$(python -c "import mem0; import os; print(os.path.dirname(mem0.__file__))") && \
    echo "Overwriting mem0 at: $MEM0_PATH" && \
    rm -rf "$MEM0_PATH"/* && \
    cp -r ./local_mem0/* "$MEM0_PATH/" && \
    echo "Files copied successfully"

# Verify our modifications are in place
RUN python -c "import mem0.configs.vector_stores.qdrant as qdrant_config; import inspect; source = inspect.getsource(qdrant_config.QdrantConfig.__init__); print('SUCCESS: Local modifications detected in qdrant.py') if 'DIAGNOSTIC:' in source else print('WARNING: Local modifications NOT found in qdrant.py')"

# Copy the application code
COPY ./openmemory/api .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
