from mangum import Mangum
from catdetec.app import app

lambda_handler = Mangum(app, lifespan="off")