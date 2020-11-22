# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-buster
#FROM debian:buster-slim


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
ADD requirements.txt .

#RUN apt-get update && apt-get install -y python3-dev python3-pip

RUN python -m pip install -r requirements.txt
#RUN python --version
#RUN pip --version
WORKDIR /app
ADD . /app

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "measure_simspeed_jsb.py"]