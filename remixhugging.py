from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelWithLMHead
import torch

class RemixTransformers():

    def __init__(self, taskType, text=None, questions=None, sequence=None, seed=None, leng=None):

        self.taskType = taskType
        print(self.taskType)
        if self.taskType == "SC":
            self.text = text
            res = self.sequenceClassification(text)
            print(res)

        elif self.taskType == "EQA":
            self.questions = questions
            # self.option = input("Multiple Questions or Single question? (Enter 1 for multiple, 2 for Single)") 
            self.context = input("Enter the context") 
            res = self.extractiveQuestionAnswering(self.context, self.questions)
            print(res)

        elif self.taskType == "MLM":
            self.sequence = sequence
            res = self.maskedLanguageModeling()
            print(res)

        # elif self.taskType == "Causal Language Modeling" or "CLM":
        #     self.causalLanguageModeling()

        elif self.taskType == "TG":
            print("In TG block")
            self.seed = seed
            self.leng = leng
            gen = self.textGeneration(seed, leng)
            print(gen)

        # elif self.taskType == "Named Entity Recognition" or "NER":
        #     self.namedEntityRecognition()

        # elif self.taskType == "Summarization" or "SUM":
        #     self.summarization()

        # elif self.taskType == "Translation" or "TRL":
        #     self.translation()

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
        nlp = pipeline("question-answering")
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

        return result

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


nlp = RemixTransformers(taskType="TG", seed="As far as I am concerned, I will", leng=50)

