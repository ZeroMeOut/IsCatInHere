FROM public.ecr.aws/lambda/python:3.10

# Set working directory
COPY predictor/ ${LAMBDA_TASK_ROOT}/predictor/

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Install dependencies 
COPY requirements.txt ${LAMBDA_TASK_ROOT}

## There should be a better way of making opencv work with lambda
ENV PATH /home/$(python3 -m site --user-base)/.local/bin:${PATH}
RUN pip install opencv-python-headless --target "${LAMBDA_TASK_ROOT}"
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"


# Set the CMD to your handler
ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD [ "lambda_function.lambda_handler" ]