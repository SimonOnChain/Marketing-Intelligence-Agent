"""AWS Lambda entrypoint via Mangum."""

from mangum import Mangum

from src.api.main import app

handler = Mangum(app, lifespan="off")

