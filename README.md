# Opinion Tree Parsing for Aspect-based Sentiment Analysis with PEFT methods


## Download repo
```
git clone --recursive https://github.com/darkqq17/Advanced_OTP.git
```

## Build Environment & Requirement

* Use conda to build the environment
```
conda env create -f environment.yml
```


## Config file

Here are some arguments of partial tuning from the configuration file that are used to build the BLIP adapter model.

* ```adapter_type: "vit"``` : vit adapter ("vit", "vit_grayscale")
     
* ```bert_adapter: "bottleneck_adapter"``` : bert adapter ("lora_adapter", or other implementations in adapterhub)



## Data preprocessing
```
python ./data/absa/process_data.py
```

## Train

```
python src/main_PEFT.py train --use-pretrained --num-epoch 250 --batch-size 256 --pretrained-model t5-base --adapter-type all
```

## Evaluation

```
python src/main_PEFT.py test --model-path ./outputs/BaseModel_res/model_dev\=0.78.pt  --test-path data/absa/lap_test.txt
```

## reference

[OpinionTreeParsing](https://github.com/HoraceXIaoyiBao/OTP4ABSA-ACL2023)


### dataset

[ACOS dataset](https://github.com/nustm/acos)
