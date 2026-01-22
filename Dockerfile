# Step 1: Use an official lightweight Python 2026-ready image
FROM python:3.12-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies for XGBoost
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your API code and the trained model
COPY . .

# Step 6: Expose the port FastAPI runs on port
EXPOSE 8000

# Step 7: Start the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# Use this in terminal to build the Image: docker build -t fraud-guard-2026 .
# Use this in terminal to run the Container: docker run -d -p 8000:8000 fraud-guard-2026