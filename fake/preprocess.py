import json
import os
from bs4 import BeautifulSoup as bs
import lxml
import re
import numpy as np

def update_progress(progress, total):
    #print("\r{}".format(progress), end="", flush=True)
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress/total * 50), progress/total*100), end="", flush=True)
    #if progress == total:
    #    print()

def indexify(articles, most_common = 5000):
    unique_words = {}
    for a in articles:
        for w in a:
            if w not in unique_words:
                unique_words[w] = 0
            else:
                unique_words[w] += 1
    counts = []
    for word, count in unique_words.items():
        counts.append((count, word))
    counts.sort(reverse=True)
    counter = 1
    indices = {}
    common = counts[:most_common]
    for _, w in common:#np.random.permutation(common):
        indices[w] = counter
        counter += 1

    res = []
    for a in articles:
        res.append([indices[w] if w in indices else 0 for w in a])
    return res, indices, unique_words

def split_and_sanitize(articles):
    res = []
    for a in articles:
        alphanum = re.sub(r'[^a-zA-Z0-9áéíĺóŕúýďľňšťžäô ]', '', a)
        res.append([x.lower() for x in alphanum.split()])
    return res

def parse_infosk(soup):
    h3 = soup.select('h1')[0]
    title = h3.text
    article = soup.find('div', id='intextad')
    return {'title': title, 'body':article.text.replace('\n',' ')}

def parse_hlavne_spravy(soup):
    h3 = soup.select('h3')[0]
    title = h3.text
    article = soup.select('.article-content')[0]
    for script in article.select('script'):
        script.decompose()
    text = article.text.strip('\n')
    return {'title': title, 'body': text}

def line_json(fname, bads=None):
    bads = []
    with open(fname, 'r') as f:
        for l in f:
            try:
                if l in ['[\n',']\n']:
                    continue
                data = json.loads(l.strip(',\n'))
                html = data['html']
                soup = bs(html, 'lxml')
                yield soup
            except KeyboardInterrupt:
                raise
            except Exception as e:
                #print(e)
                bads.append(l)
                #print(':(', len(bads))
                pass

def process_infosk(num_of_samples = 25000):
    articles = []
    i = 0
    for l in line_json('./data/info_sk.json'):
        if i == num_of_samples:
            break
        p = parse_infosk(l)
        articles.append(p['body'])
        i += 1
        update_progress(i, num_of_samples)
    print()
    return split_and_sanitize(articles)

def process_hlavnespravy(num_of_samples = 25000):
    articles = []
    i = 0
    for l in line_json('./data/hlavnespravy.json'):
        if i == num_of_samples:
            break
        p = parse_hlavne_spravy(l)
        articles.append(p['body'])
        i += 1
        update_progress(i, num_of_samples)
    print()
    return split_and_sanitize(articles)

infosk = process_infosk()
hlavne = process_hlavnespravy()
split = len(infosk)
indexed, dic, _ = indexify(infosk + hlavne, most_common=10000)
infosk = indexed[:split]
hlavne = indexed[split:]
f = open('./data/hlavne_indexed.json', 'w')
json.dump(hlavne, f, indent=4, ensure_ascii=False)
f.close()
f = open('./data/info_indexed.json', 'w')
json.dump(infosk, f, indent=4, ensure_ascii=False)
f.close()
f = open('./data/wordindexing.json', 'w')
json.dump(dic, f, indent=4, sort_keys=True, ensure_ascii=False)
f.close()
f = open('./data/dataset.json', 'w')
dataset = np.random.permutation([[x, 0] for x in infosk] + [[x, 1] for x in hlavne]).tolist()
json.dump(dataset, f, indent=4, ensure_ascii=False)
f.close()