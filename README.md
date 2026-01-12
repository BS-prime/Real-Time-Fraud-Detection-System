# Real_Time_Fraud_Detection_System
This project is about detecting credit card fraud in real time.


Step 1: "Setup a Python script to ""produce"" fake transactions into a Kafka topic."


Step 2: "Write a ""consumer"" script that reads the stream and calculates rolling averages."


Step 3: Train a LightGBM model on a static version of the fake data.


Step 4: Deploy the model as an API using FastAPI to score incoming stream events.


Step 5: "Build a Streamlit dashboard to visualize ""Live Fraud vs. Legitimate"" counts."