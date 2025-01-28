# track2-final-assessment
🟢 genai track 2 final assessment reference code

## Description

## Installation

Install dependencies 

```bash
pip install -r requirements.txt
```

Start vector database (chromaDB) using docker-compose

```bash
docker compose up -d
```

Run the code 

```bash
python src/main.py
```

## Testing 

Ping the ChromaDB container to check if it is running

```bash
curl localhost:8000/api/v1/heartbeat
```