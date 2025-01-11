FROM public.ecr.aws/lambda/python:3.10

# Set working directory
COPY catdetec/ ${LAMBDA_TASK_ROOT}/catdetec/

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
ENV PATH /home/$(python3 -m site --user-base)/.local/bin:${PATH}

RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"


# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]