"""
HappyBERT: a wrapper over PyTorch's BERT implementation

"""

# disable pylint TODO warning
# pylint: disable=W0511
import re
from transformers import (
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForQuestionAnswering,
    BertTokenizer
)
from transformers import DistilBertTokenizerFast
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AdamW
from happytransformer.happy_transformer import HappyTransformer

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

class HappyBERT(HappyTransformer):
    """
    Currently available public methods:
        BertForMaskedLM:
            1. predict_mask(text: str, options=None, k=1)
        BertForSequenceClassification:
            1. init_sequence_classifier()
            2. advanced_init_sequence_classifier()
            3. train_sequence_classifier(train_csv_path)
            4. eval_sequence_classifier(eval_csv_path)
            5. test_sequence_classifier(test_csv_path)
        BertForNextSentencePrediction:
            1. predict_next_sentence(sentence_a, sentence_b)
        BertForQuestionAnswering:
            1. answer_question(question, text)

            """

    def __init__(self, model='bert-base-uncased'):
        super().__init__(model, "BERT")
        self.mlm = None  # Masked Language Model
        self.nsp = None  # Next Sentence Prediction
        self.qa = None   # Question Answering
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.masked_token = self.tokenizer.mask_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token

    def _get_masked_language_model(self):
        """
        Initializes the BertForMaskedLM transformer
        """
        self.mlm = BertForMaskedLM.from_pretrained(self.model)
        self.mlm.eval()

    def read_dataset(self,path):
        f=pd.read_excel(path, engine='openpyxl')
        con1= f['context'].tolist()
        a1= f['time-start'].tolist()
        a2= f['time-end'].tolist()
        a3= f['date-start'].tolist()
        a4= f['date-end'].tolist()
        a5= f['user-start'].tolist()
        a6= f['user-end'].tolist()
        q=["What time is the appointment?","what calendar date is the appointment?","who is the appointment with?"]
        contexts = []
        questions = []
        answers = []
        for i,c in enumerate(con1):
            contexts.append(c)
            questions.append(q[0])
            answers.append({"answer_start":a1[i],"answer_end":a2[i]})
            contexts.append(c)
            questions.append(q[1])
            answers.append({"answer_start":a3[i],"answer_end":a4[i]})
            contexts.append(c)
            questions.append(q[2])
            answers.append({"answer_start":a5[i],"answer_end":a6[i]})

        return contexts, questions, answers

    def add_token_positions(self,encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    def train(self):
        self.model=BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        train_contexts,train_questions,train_answers = self.read_dataset("dataset1.xlsx")
        val_contexts,val_questions,val_answers = self.read_dataset("valdataset1.xlsx")
        train_encodings = self.tokenizer(train_contexts, train_questions, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_contexts, val_questions, truncation=True, padding=True)
        self.add_token_positions(train_encodings, train_answers)
        self.add_token_positions(val_encodings, val_answers)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(device)
        self.model.train()


        train_dataset = Dataset(train_encodings)
        val_dataset = Dataset(val_encodings)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        optim = AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(3):
            print(epoch)
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optim.step()

        self.model.eval()

    def _get_next_sentence_prediction(self):
        """
        Initializes the BertForNextSentencePrediction transformer
        """
        self.nsp = BertForNextSentencePrediction.from_pretrained(self.model)
        self.nsp.eval()

    def _get_question_answering(self):
        """
        Initializes the BertForQuestionAnswering transformer
        NOTE: This uses the bert-large-uncased-whole-word-masking-finetuned-squad pretraining for best results.
        """
        self.qa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.qa.eval()

    def predict_next_sentence(self, sentence_a, sentence_b):
        """
        Determines if sentence B is likely to be a continuation after sentence
        A.
        :param sentence_a: First sentence
        :param sentence_b: Second sentence to test if it comes after the first
        :return tuple: True if b is likely to follow a, False if b is unlikely
                       to follow a, with the probabilities as the second item
                       of the tuple
        """

        if not self.__is_one_sentence(sentence_a) or not  self.__is_one_sentence(sentence_b):
            self.logger.error("Each inputted text variable for the \"predict_next_sentence\" method must contain a single sentence")
            exit()

        if self.nsp is None:
            self._get_next_sentence_prediction()
        connected = sentence_a + ' ' + sentence_b
        tokenized_text = self._get_tokenized_text(connected)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = self._get_segment_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            predictions = self.nsp(tokens_tensor, token_type_ids=segments_tensors)[0]

        if predictions[0][0] >= predictions[0][1]:
            return True
        return False

    def __is_one_sentence(self, text):
        """
        Used to verify the proper input requirements for sentence_relation.
        The text must contain no more than a single sentence.
        Casual use of punctuation is accepted, such as using multiple exclamation marks.
        :param text: A body of text
        :return: True if the body of text contains a single sentence, else False
        """
        split_text = re.split('[?.!]', text)
        sentence_found = False
        for possible_sentence in split_text:
            for char in possible_sentence:
                if char.isalpha():
                    if sentence_found:
                        return False
                    sentence_found = True
                    break
        return True

    def answer_question(self, question, text):
        """
        Using the given text, find the answer to the given question and return it.

        :param question: The question to be answered
        :param text: The text containing the answer to the question
        :return: The answer to the given question, as a string
        """
        if self.qa is None:
            self._get_question_answering()
        input_text = self.cls_token + " " + question + " " + self.sep_token + " " + text + " " + self.sep_token
        input_ids = self.tokenizer.encode(input_text)
        sep_val = self.tokenizer.encode(self.sep_token)[-1]
        token_type_ids = [0 if i <= input_ids.index(sep_val) else 1
                          for i in range(len(input_ids))]
        token_tensor = torch.tensor([input_ids])
        segment_tensor = torch.tensor([token_type_ids])
        with torch.no_grad():
            scores= self.qa(input_ids=token_tensor,
                                               token_type_ids=segment_tensor)
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        answer_list = all_tokens[torch.argmax(scores[0]):
                                 torch.argmax(scores[1])+1]
        answer = self.tokenizer.convert_tokens_to_string(answer_list)
        answer = answer.replace(' \' ', '\' ').replace('\' s ', '\'s ')
        return answer
