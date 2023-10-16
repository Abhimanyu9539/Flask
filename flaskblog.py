from flask import Flask
from flask import render_template, url_for

app = Flask(__name__)

POSTS = [
    {
        'author':"ABD",
        "title":"Blog 1",
        "content":"first blog post",
        "date":'Oct 16, 2023'
    },
        {
        'author':"ABD",
        "title":"Blog 2",
        "content":"second blog post",
        "date":'Oct 17, 2023'
    }
]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts = POSTS)


@app.route("/about")
def about():
    return render_template('about.html', title="about")

if __name__ == "__main__":
    app.run(debug=True)