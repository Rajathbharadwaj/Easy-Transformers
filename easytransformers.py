from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelWithLMHead, AutoModelForTokenClassification
import torch

class EasyTransformers():

    def __init__(self, taskType, text=None, context=None, questions=None, sequence=None, seed=None, leng=None, nerSequence = None, sentence=None, translate = None):
        """
        Essentially makes usage of 🤗 Transformers easy, plug & play like.

        Usage:
        taskType : Type of the task you want to perform -> {"Sequence Classification":"SC", 
        "Extractive Question Answering":"EQA",
        "Masked Language Modeling":"MLM",
        "Text Generation":"TG",
        "Summarization":"SUM",
        "Translation":"TLR"}

        
        
        """
        self.taskType = taskType
        print(self.taskType)
        if self.taskType == "SC":
            print("[INFO....] Sequence Classification")
            self.text = text
            res = self.sequenceClassification(text)
            print(res)

        elif self.taskType == "EQA":
            print("[INFO....] Extractive Question Answering")
            self.questions = questions
            self.context = context
            res = self.extractiveQuestionAnswering(self.context, self.questions)
            print("Answers\n")
            print(res)

        elif self.taskType == "MLM":
            print("[INFO....] Masked Language Modeling")
            self.sequence = sequence
            res = self.maskedLanguageModeling()
            print(res)


        elif self.taskType == "TG":
            print("[INFO....] Text Generation")
            self.seed = seed
            self.leng = leng
            gen = self.textGeneration(seed, leng)
            print("Generated text " + gen)

        elif self.taskType == "NER":
            print("[INFO....] Name Entity Recognition")
            self.nerSeq = nerSequence
            self.namedEntityRecognition(self.nerSeq)

        elif self.taskType == "SUM":
            print("[INFO....] Summarization")
            self.sent = sentence
            summary =  self.summarization(self.sent)
            print(summary)

    def sequenceClassification(self, text):
        nlp = pipeline("sentiment-analysis")
        result = nlp(text)[0]
        return f"label: {result['label']}, with score: {round(result['score'], 4)}"
    
    def extractiveQuestionAnswering(self, context, questions):
        
        print("[INFO....] Downloading bert for you")
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        print("[INFO....] Downloaded bert for you")
        result = []
        ans = {}
        context = r"""{}""".format(context)
        for quest in questions:
            inputs = tokenizer(quest, context, add_special_tokens=True, return_tensors="pt")
            inputIds = inputs["input_ids"].tolist()[0]
        
            text_tokens = tokenizer.convert_ids_to_tokens(inputIds)
            answer_start_scores, answer_end_scores = model(**inputs)
            answer_start = torch.argmax(
                answer_start_scores
            )
            answer_end = torch.argmax(answer_end_scores) + 1

            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputIds[answer_start:answer_end]))
            result.append(answer)
        for i, q in enumerate(questions):
            ans[q] = result[i]
        return ans

    def maskedLanguageModeling(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
        sequence = "{}".format(self.sequence)
        sequence = sequence.replace("{}", "{tokenizer.mask_token}")
        inpt = tokenizer.encode(sequence, return_tensors='pt')
        mask_token_index = torch.where(inpt == tokenizer.mask_token_id)[1]
        token_logits = model(inpt).logits
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        tokens = []
        for token in top_5_tokens:
            tokens.append(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
        return tokens

    
    def textGeneration(self, seed, length):
        text_gen = pipeline("text-generation")
        generated = text_gen(seed, max_len = length, do_sample = False)
        return generated

    def namedEntityRecognition(self, nerSeq):
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        label_list = [
             "O",       # Outside of a named entity
             "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
             "I-MISC",  # Miscellaneous entity
             "B-PER",   # Beginning of a person's name right after another person's name
             "I-PER",   # Person's name
             "B-ORG",   # Beginning of an organisation right after another organisation
             "I-ORG",   # Organisation
             "B-LOC",   # Beginning of a location right after another location
             "I-LOC"    # Location
        ]
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(nerSeq)))
        inputs = tokenizer.encode(nerSeq, return_tensors="pt")
        outputs = model(inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        result = {}
        for token, prediction in zip(tokens, predictions[0].numpy()):
            result[token] = label_list[prediction]
        
        return result
    
    def summarization(self, sentence):
        summary = pipeline("summarization")
        summaryed = summary(sentence, max_length = 130, do_sample=False)
        return summaryed


    