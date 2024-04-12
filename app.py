from flask_cors import CORS
from flask import Flask, request, render_template
import sys
from llm import llm_ans_question

app = Flask(__name__, template_folder='template')
CORS(app)  # Enable CORS for all routes in the app


@app.route('/')
def start():
    return render_template("index.html")


@app.route('/api/v0/ask', methods=['GET'])
def generate_ans():
    user_query = request.args.get('question')
    if user_query:
        return llm_ans_question(user_query)
    else:
        return "No question provided in the request."


if __name__ == '__main__':
    app.run()
