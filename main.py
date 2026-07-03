import sys, logging
from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes.transformers import app as transformers_app
from domain.auth import app as auth_app

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='[%(levelname)s] %(message)s')

app = FastAPI(title="Secure Clinical Documentation Pipeline")

app.include_router(auth_app.router, prefix='/api/v1')
app.include_router(transformers_app.router, prefix='/api/v1')
