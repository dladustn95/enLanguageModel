# enLanguageModel

## Installation

transformer 라이브러리 코드를 직접 수정하므로 가상환경 사용을 권장

```bash
python -m venv .enLM
. .enLM/bin/activate
pip install -r requirements.txt
```
transformer_file 디렉토리에 있는 파일을 .enLM/lib/python3.7/site-packages/transformers/ 로 이동

가상환경은 deactivate 명령어로 종료

## 학습 방법

train.py / train_key.py 파일 실행 (키워드의 사용 유무)

train의 argument:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset.
use_adapter | `bool` | `True` | Use adapter or not
keyword_module | `str` | `""` | Use keyword module or not
train_batch_size | `int` | `20` | Batch size for training
valid_batch_size | `int` | `20` | Batch size for validation
gradient_accumulation_steps | `int` | `8` | Accumulate gradients on several steps
lr | `float` | `6.25e-5` | Learning rate
max_norm | `float` | `1.0` | Clipping gradient norm
n_epochs | `int` | `5` | Number of training epochs
personality_permutations | `int` | `1` | Number of permutations of personality sentences
device | `str` | `"cuda" if torch.cuda.is_available() else "cpu"` | Device (cuda or cpu)
fp16 | `str` | `""` | Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)
local_rank | `int` | `-1` | Local rank for distributed training (-1: not distributed)
gpt2_model_name | `str` | `"gpt2"` | Path, url or short name of the model

```bash
python train.py --dataset_path DATAPATH/Name
```
 
## 문장 생성 방법

interact.py / interact_key.py 파일 실행 (키워드의 사용 유무)

interact의 argument:

Argument | Type | Default value | Description
---------|------|---------------|------------
dataset_path | `str` | `""` | Path or url of the dataset.
use_adapter | `bool` | `True` | Use adapter or not
keyword_module | `str` | `""` | Use keyword module or not
model_checkpoint | `str` | `""` | Path, url or short name of the model
device | `str` | `cuda` if `torch.cuda.is_available()` else `cpu` | Device (cuda or cpu)
gpt2_model_name | `str` | `"gpt2"` | name of the model ex)openai-gpt
no_sample | action `store_true` | Set to use greedy decoding instead of sampling
max_length | `int` | `40` | Maximum length of the output utterances
min_length | `int` | `1` | Minimum length of the output utterances
seed | `int` | `0` | Seed
temperature | `int` | `0.7` | Sampling softmax temperature
top_k | `int` | `0` | Filter top-k tokens before sampling (`<=0`: no filtering)
top_p | `float` | `0.9` | Nucleus filtering (top-p) before sampling (`<=0.0`: no filtering)

```bash
python interact.py --dataset_path DATAPATH/Name --model_checkpoint MODELPATH/
```

## 데이터 포맷

Source|Target 형태로 txt파일 구성.

아래의 형태로 같은 경로에 데이터가 존재해야 함.  
Name_train.txt  / Name_train_keyword.txt  
Name_valid.txt  / Name_valid_keyword.txt  
Name_test.txt   / Name_test_keyword.txt  
