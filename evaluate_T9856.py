import os
import numpy as np
from numpy.linalg import norm
import scipy.stats

def cosine_similarity(a, b):
    ret = np.inner(a, b) / (norm(a) * norm(b)) 
    return 0 if np.isnan(ret) else ret 

# load all embeddings
def load_embeddings(folder, embeddings={}):
    for filename in os.listdir(folder):
        if filename in ['.', '..']: continue
        bundle = os.path.splitext(filename)[0].replace('_', ' ')
        emb = np.load(os.path.join(folder, filename), allow_pickle=True)
        embeddings[bundle] = emb
    return embeddings

if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('eval_file', help='evaluation file')
        parser.add_argument('folder', help='embedding folder')
        return parser.parse_args()
    args = parse_args()

    print('loading embeddings...')
    embeddings = load_embeddings(args.folder)
    print(f' #{len(embeddings)} embeddings is loaded.')

    eval_data = [tuple(line.strip().split(',')) for line in open(args.eval_file, 'r')]
    print(f' #{len(eval_data)} eval data loaded')
    
    print('calculating similarities')
    our_similarities = []
    target_scores = []
    fail_list = []
    for p1, p2, score in eval_data:
        try: 
            emb1 = np.load(os.path.join(args.folder, p1.replace(' ', '_')+'.npy'), allow_pickle=True)
            emb2 = np.load(os.path.join(args.folder, p2.replace(' ', '_')+'.npy'), allow_pickle=True)
        except:
            fail_list.append((p1, p2))
            continue
        our_similarities.append(cosine_similarity(emb1, emb2))
        target_scores.append(score)
    print(f'finished. #failure={len(fail_list)}')
    print(f' our length={len(our_similarities)}; target length={len(target_scores)}')

    # log
    with open('simlarity.log', 'w+') as f:
        for s in our_similarities:
            f.write(str(s)) 
            f.write('\n')

    print('calculating result...')
    our_similarities = np.array(our_similarities, dtype='float')
    target_scores = np.array(target_scores, dtype='float')
    pearsonr = scipy.stats.pearsonr(our_similarities, target_scores)
    spearmanr = scipy.stats.spearmanr(our_similarities, target_scores)
    print(f'pearson coefficient: {pearsonr[0]:5.3f} (p-value={pearsonr[1]:.3e})')
    print(f'spearman coefficient: {spearmanr[0]:5.3f} (p-value={spearmanr[1]:.3e})')

