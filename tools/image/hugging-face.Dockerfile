FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 7860
CMD ["python", "-m", "ai_agent.app"]