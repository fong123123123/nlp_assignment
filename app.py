import PySimpleGUI as sg
from transformers import BertForQuestionAnswering, BertTokenizer
import torch


# Load the pre-trained BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Load your corpus or knowledge base
# Replace this with your own code to load the text data
corpus = "a is Looi, b is Fong, Fong is 10 year old, Looi is 12 year old, Fong love to eat bing chilling, Looi love to eat laoganma"

# Initialize the conversation history
conversation_history = []

# Define the layout for the GUI
layout = [
    [sg.Text('Enter your question:')],
    [sg.InputText(key='question')],
    [sg.Button('Submit'), sg.Button('Exit')],
    [sg.Text('', size=(60, 20), key='output')],
    [sg.Text('Conversation History:')],
    [sg.Multiline(size=(60, 10), key='history', disabled=True)]
]

# Create the window
window = sg.Window('Question Answering System', layout)

# Event loop
while True:
    event, values = window.read()

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    elif event == 'Submit':
        question = values['question']

        # Tokenize the question and corpus
        question_tokens = tokenizer.encode(question, add_special_tokens=True)
        corpus_tokens = tokenizer.encode(corpus, add_special_tokens=True)

        # Run the BERT model to get the start and end positions of the answer
        input_ids = torch.tensor([question_tokens + corpus_tokens[1:]])
        output = model(input_ids)
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits) + 1

        # Decode the answer tokens
        answer_tokens = corpus_tokens[answer_start:answer_end + 1]
        answer = tokenizer.decode(answer_tokens)

        # Update the conversation history
        conversation_history.append(f"Question: {question}")
        conversation_history.append(f"Answer: {answer}")

        # Display the answer and update the conversation history
        window['output'].update(answer)
        window['history'].update('\n'.join(conversation_history))

        # Clear the input field
        window['question'].update('')

# Close the window
window.close()