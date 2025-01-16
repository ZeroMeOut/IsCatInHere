from mangum import Mangum
from predictor.main import app

lambda_handler = Mangum(app, lifespan="off")