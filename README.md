# Welcome to Easy Transformers!
   Essentially makes usage of 🤗 Transformers easy, plug & play like.

# Files

easytransformers.py -> Contains the implementation of 🤗 Transformers, does all the heavy lifting so you  don't have to write much code.


## Usage  
    from easytransformers import  EasyTransformers
    #instantiate EasyTransformers
    nlp = EasyTransformers(taskType, text, context, questions, sequence, seed, leng, nerSequence, sentence, translate)

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

