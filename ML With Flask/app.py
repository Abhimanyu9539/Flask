# Import necessary packages
from flask import Flask, render_template, redirect, url_for, request


# Initialize the flask web app
app = Flask(__name__)

# Initialize the home route
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/success/<int:marks>')
def success(marks):
    return render_template("pass.html", score=marks)

@app.route('/fail/<int:marks>')
def fail(marks):
    return render_template("fail.html", score = marks)

@app.route("/result/<int:score>")
def result(score):
    res = ""
    if score>52:
        res = "success"
    else:
        res = "fail"
    return redirect(url_for(res, marks=score))

## This code handles the form submission from the submit page and calculates the total score. 
## The score is then passed to the result page.
@app.route("/submit", methods = ['POST', "GET"])
def submit():
    total_score = 0
    if request.method == "POST":
        science = float(request.form['science'])
        maths = float(request.form['maths'])
        c = float(request.form['c'])
        datascience = float(request.form['datascience'])

        total_score = (science + maths + c + datascience) / 4
        print(total_score)
    
    return redirect(url_for('result', score = total_score))


#Run the flask app
if __name__ == "__main__":
    app.run(debug=True)