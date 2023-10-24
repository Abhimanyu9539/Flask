from flask import Flask
from flask_sqlalchemy import SQLAlchemy


import os
basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flask_app.db'
db = SQLAlchemy(app)

from flaskblog import routes