FROM public.ecr.aws/lambda/python:3.10

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy function code
COPY app.py .
COPY best.pt .

# Set the CMD to your handler
CMD [ "app.handler" ]