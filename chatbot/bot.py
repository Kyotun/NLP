from chatterbot import ChatBot, filters
from chatterbot.trainers import ListTrainer, ChatterBotCorpusTrainer

chatbot = ChatBot(
    'Charlie'
)
exit_conditions = (":q", "quit", "exit")

while True:
    query = input("> ")
    if query in exit_conditions:
        break
    print(f"🪴 {chatbot.get_response(query)}")
