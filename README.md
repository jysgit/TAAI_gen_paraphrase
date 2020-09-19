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
如果想利用以不同參數依序訓練模型，可參考 `train_all.sh` 這個script來進行。   

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

## InferSent encoding
1. 下載 InferSent pretrained model：  
   ```
   mkdir encoders
   curl -Lo encoders/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
   ```
2. 利用 get_emb_from_InferSent.py 抓取 phrase embeddings。  
   ex. `python get_emb_from_InferSent.py -w2v models/${YOUR_MODEL}/vector.kv -if encoders/infersent2.pkl -wd word_embs -bd bundle_embs bundles.txt`  
   **參數說明：**  
    + `-w2v`: gensim.KeyedVector 格式的 w2v embedding 位置。可用 [train.py](#Training) 輸出的 `vector.kv` 檔案作為 input。
    + `-if`: InferSent 的 pretrained model 位置。
    + `-wd`: word embedding 存放位置。內部需有 InferSent 的 beggin-of-sent 與 end-of-sent 兩個 embedding 存在。
    + `-bd`: 輸出的 bundle embedding 存放位置。

## Compare embedding similarities
`most_similar.[py|ipynb]`為方便快速查詢 embedding 最相近的 phrases 的互動式程式，可用來找尋指定資料夾中最相近的 bundles。  
 + 用法： `python most_similar.py -n ${#RESULT} ${EMBEDDING_FOLDER} [${EMBEDDING_FOLDER}...]`  
 + 範例： `python most_similar.py -n 10 bundle_emb1/ budle_emb2/`  

## Evaluate with T9856 dataset
* Original Dataset: [From IBM](https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml)  

1. 先取得所有 evaluation phrase 的 embedding:  
   + InferSent: `python get_emb_from_InferSent.py -w2v models/${YOUR_MODEL}/vector.kv -if encoders/infersent2.pkl -wd word_embs -bd eval_embs data/T9856_phrases.txt`
2. 算出我們 model 與以 T9856 作為 ground truth 的相關係數  
   `python evaluate_T9856.py data/T9856_eval_file.txt eval_embs`
