# TAAI_gen_paraphrase
Use lexical bundles in "Lonely Planet" to generate comprehensive phrase embedding and find paraphrases.

## Word2Vec Training

### Training From scratch
可利用train.py來進行Word2Vec模型訓練。  
`python train.py -i ${TRAINING_DATA} -o ${OUTPUT_PATH}`
可指定參數：  
 + `-e`: training epochs  
 + `-d`: embedding dimension  
 + `-w`: window size
 + 更多參數可參考 `python train.py -h`
如果想利用多參數訓練不同模型，可參考 `train_all.sh` 這個script來進行。   

#### Output 
若有指定 output 位置，則資料夾內會產生三或四個檔案：
1. `model`: gensim.Word2Vec 格式，可用以接續訓練或進行 finetune。
2. `loss`: training 途中每個 ephoch 的loss 檔案，一行一個紀錄。
3. `vector.kv`: gensim.KeyedVector 格式，可用以取得此 model 的 word embbeding 。
4. (若有指定`--eval`) `accuracy`: 利用 eval file 進行 eval 的結果紀錄，第一行為最終 accuracy ，第二行為每個 task 的詳細結果紀錄。

### Finetuning 
 1. 下載 [GoogleNews pretrained model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)，並放到 models/ 下
    ```
    mkdir models
    gdown -O models/GoogleNews-vectors-negative300.bin --id 0B7XkCwpI5KDYNlNUTTlSS21pQmM
    ```
 2. 於訓練時指定pretrained model位置：`python train.py -i ${INPUT} -p models/GoogleNews-vectors-negative300.bin`（其餘參數可參照上方說明）。
