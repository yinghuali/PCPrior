FROM python:3.9.12
WORKDIR ./PCPrior
ADD . .
RUN pip install -r requirements.txt
CMD ["python3"]
