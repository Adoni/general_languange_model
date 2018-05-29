# general_languange_model
A general language model

## Usage

First, you need to train a model

```
python language_model/train.py --data_path data_path --model_path model_path
```

`model_path` is where we gonna save our model.
If there is already some file in that path, we gonna load this model first.

`data_path` is the path to our data.
There should at least a text file called `train.txt`.
You can also add `valid.txt` and `test.txt`

When we finish training, the model should be saved in the `model_path` with two files:
`model.pkl` and `dictionary.pkl`.

Now we can use it online.

```
from language_model import language_model as lm

lm.LanguageModel(model_path)
sentences=[
    "This is a good sentence.",
    "Sentence good is a this.",
]
for s in sentences:
    print(s)
    print(lm.scoring(s))
```



## Incremental training


