# Welcome to Easy Transformers!
   Essentially makes usage of 🤗 Transformers easy, plug & play like.

# Files

easytransformers.py -> Contains the implementation of 🤗 Transformers, does all the heavy lifting so you  don't have to write much code. 
This uses PyTorch, so PyTorch is needed.

# What Easy Transformers can do?


Basically lets you do NLP, in essentially a single line of code. It uses 🤗 Transformers. <br>
Supported as of now: <br>
    "Sequence Classification" : "SC", <br> 
    "Extractive Question Answering" : "EQA",<br>
    "Masked Language Modeling" : "MLM",<br>
    "Text Generation" : "TG",<br>
    "Summarization" : "SUM"

# What is: 

## Sequence Classification (SC)
    Sequence classification is the task of classifying sequences according to a given number of classes.

## Extractive Question Answering (EQA)
    Extractive Question Answering is the task of extracting an answer from a text given a question.

## Masked Language Modeling (MLM)
    Masked language modeling is the task of masking tokens in a sequence with a masking token, and prompting the model to fill that mask with an appropriate token.

## Text Generation (TG)
    In text generation (a.k.a open-ended text generation) the goal is to create a coherent portion of text that is a continuation from the given context.

## Named Entity Recognition (NER)
    Named Entity Recognition (NER) is the task of classifying tokens according to a class, for example, identifying a token as a person, an organisation or a location.

## Summarization (SUM)
    Summarization is the task of summarizing a document or an article into a shorter text.

## Usage  
    from easytransformers import  EasyTransformers
    #instantiate EasyTransformers
    nlp = EasyTransformers(taskType, text, context, questions, sequence, seed, leng, nerSequence, sentence)
-----------------

## Demo

	This demo shows the usage for Extractive Question Answering (EQA)

    context =  "🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch."
    questions = ["How many pretrained models are available in 🤗 Transformers?","What does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?"]
    
    sc = EasyTransformers(taskType="EQA", questions=questions, context=context)

## Output
	[INFO....] Extractive Question Answering 
	[INFO....] Downloading bert for you 
    [INFO....] Downloaded bert for you 
	Answers 
	{'How many pretrained models are available in 🤗 Transformers?': 'over 32 +', 'What does 🤗 Transformers provide?': 'general - purpose architectures', '🤗 Transformers provides interoperability between which frameworks?': 'tensorflow 2 . 0 and pytorch'}


| Args | Used in |
|--|--|
|**text** : str|  Sequence Classification **[not needed for everything else]**|
|**context** : str, **question** : list| Extractive Question Answering **[not needed for everything else]** |
|**sequence** : str [ Mask with **{ }** ]| Masked Language Modeling **[not needed for everything else]** |
| **seed**:str, **leng** : int [length to generate] | Text Generation **[not needed for everything else]** |
| **nerSequence** : str | Named Entity Recognition **[not needed for everything else]** |
| **sentence** : str | Summarization **[not needed for everything else]** |


# Citing

    @article{Wolf2019HuggingFacesTS,
      title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
      author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
      journal={ArXiv},
      year={2019},
      volume={abs/1910.03771}
    }

