""" Main application file for the Medical RAG Chatbot. """

import os
import markdown
from flask import (
    Flask, render_template, request, 
    session, redirect, url_for
)
from src.components.retriever import create_qa_chain
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            # 1. Update session with user message immediately
            session["messages"].append({"role": "user", "content": user_input})
            session.modified = True

            try:
                # 2. Re-create the chain (or load it from a global variable if optimized)
                qa_chain = create_qa_chain()
                
                # 3. Convert session history to LangChain Message objects
                chat_history = []
                for msg in session["messages"][:-1]: # Exclude the current user message
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))

                # 4. Invoke the chain with input AND history
                response = qa_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                # 5. Extract the answer
                answer = response.get("answer", "No response generated.")

                # 6. Convert to HTML and save
                formatted_html = markdown.markdown(answer, extensions=['fenced_code', 'tables'])
                session["messages"].append({"role": "assistant", "content": formatted_html})
                session.modified = True

            except Exception as e:
                error_msg = f"Error in chain: {str(e)}"
                return render_template("index.html", messages=session["messages"], error=error_msg)
            
        return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))

@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)