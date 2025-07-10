from flask import Flask, render_template, request, jsonify
from models import LLMService, db, ChatHistory, dummy_llm_service

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
db.init_app(app)
app.config["LLM_SERVICE"] = LLMService("Ты оператор техподдержки, отвечай вежливо")


@app.route("/")
def home():
    chat_history = ChatHistory.query.order_by(ChatHistory.timestamp.asc()).all()
    return render_template("index.html", chat_history=chat_history)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    llm_reply = app.config["LLM_SERVICE"].chat(user_message)
    # Save to chat history
    new_entry = ChatHistory(user_message=user_message, llm_reply=llm_reply)
    db.session.add(new_entry)
    db.session.commit()
    return jsonify({"reply": llm_reply})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
