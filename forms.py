from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo


class RegistrationForm(FlaskForm):
    # Username field
    username = StringField('Username',
                           validators=[DataRequired(), Length(min=2, max=20)])

    # Email field
    email = StringField('Email',
                        validators=[DataRequired(), Email()])

    # Password field
    password = PasswordField('Password', validators=[DataRequired()])

    # Confirm password field
    confirm_password = PasswordField('Confirm Password',
                                     validators=[DataRequired(), EqualTo('password')])

    # Submit button
    submit = SubmitField('Sign Up')


class LoginForm(FlaskForm):
    email = StringField('Email',
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')