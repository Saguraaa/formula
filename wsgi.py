"""Linux application entry point; use one worker to share the inference lock."""
import os
from web_app import create_app

app = create_app(device=os.environ.get('FORMULA_DEVICE', 'cpu'))
