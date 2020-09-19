import os
import numpy as np
from numpy.linalg import norm


# --- utilities --- # 

def head(li, n=10):
    for i, content in enumerate(li):
        if i > n: break
        print(content)

def cosine_similarity(a, b):
    ret = np.inner(a, b) / (norm(a) * norm(b))
    return 0 if np.isnan(ret) else ret

def most_similar(target, bundles, n=5):
    similarities = []
    target_emb = bundles[target]
    for bundle, bundle_emb in bundles.items():
        if bundle == target: continue
        similarities.append((target, bundle, cosine_similarity(target_emb, bundle_emb)))
    similarities.sort(key=lambda emb:-emb[2])
    return similarities[:n]

def print_similarity(tuples):
    head = True
    for t in tuples:
        if head:
            print(f'{t[0]}')
            head = False
        print(f'  > {t[1]}\t{t[2]}')

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
        parser.add_argument('folders', nargs='+', help='embedding folders')
        parser.add_argument('-n', default=5, type=int, help='how many similar bundles to show')
        return parser.parse_args()
    args = parse_args()

    print('loading embeddings...')
    embeddings = {}
    for folder in args.folders:
        embeddings = load_embeddings(folder, embeddings)
    print(f' #{len(embeddings)} embeddings is loaded.')
    print('Enter quit or q to exit.')

    while True:
        print()
        query = input(' >> input: ')
        if query in ['quit', 'q']: break
        try:
            print_similarity((most_similar(query, embeddings, args.n)))
        except:
            print('Query not found!')

    print()
    print('quit')
