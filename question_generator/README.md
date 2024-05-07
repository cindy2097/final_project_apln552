# question_generator

Question Generator is an NLP system for generating reading comprehension-style questions from texts such as news articles or pages excerpts from books. The system is built using pretrained models from [HuggingFace Transformers](https://github.com/huggingface/transformers). There are two models: the question generator itself, and the QA evaluator which ranks and filters the question-answer pairs based on their acceptability.


# Training and evaluating

python question_generator/training/qg_train.py

python question_generator/training/qa_eval_train.py

## Usage

The easiest way to generate some questions is to clone the github repo and then run `qg_run.py` like this:

```
git clone https://github.com/amontgomerie/question_generator
cd question_generator
pip install -r requirements.txt
python run_qg.py --text_file articles/slp_ch2.txt
```

This will generate 20 question-answer pairs of mixed style (full-sentence and multiple choice) based on the article specified in `--text_file` and print them to the console.

The `QuestionGenerator` class can also be instantiated and used like this:

```python
from questiongenerator import QuestionGenerator
qg = QuestionGenerator()
qg.generate(text, num_questions=20)
```

This will generate 10 questions of mixed style and return a list of dictionaries containing question-answer pairs. In the case of multiple choice questions, the answer will contain a list of dictionaries containing the answers and a boolean value stating if the answer is correct or not. The output can be easily printed using the `print_qa()` function. For more information see the question_generation_example notebook.

### Choosing the number of questions

The desired number of questions can be passed as a command line argument using `--num_questions` or as an argument when calling `qg.generate(text, num_questions=20`. If the chosen number of questions is too large, then the model may not be able to generate enough. The maximum number of questions will depend on the length of the input text, or more specifically the number of sentences and named entities containined within text. Note that the quality of some of the outputs will decrease for larger numbers of questions, as the QA Evaluator ranks generated questions and returns the best ones.

### Answer styles

The system can generate questions with full-sentence answers (`'sentences'`), questions with multiple-choice answers (`'multiple_choice'`), or a mix of both (`'all'`). This can be selected using the `--answer_style` or `qg.generate(answer_style=<style>)` arguments.


### QA Evaluator

The QA evaluator takes a question answer pair as an input and outputs a value representing its prediction about whether the input was a valid question and answer pair or not. The model is `bert-base-cased` with a sequence classification head. The pretrained model was finetuned on the same data as the question generator model. The question and answer were concatenated 50% of the time. In the other 50% of the time a corruption operation was performed (either swapping the answer for an unrelated answer, or by copying part of the question into the answer). 
```
