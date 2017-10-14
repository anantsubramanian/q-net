import argparse
import hashlib
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-a', required=True)
    parser.add_argument('--output-b', required=True)
    parser.add_argument('--b-percentage', type=int, required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as fileobj:
        articles = json.loads(fileobj.read())

    a_articles = []
    b_articles = []
    for article in articles:
        hash_mod_100 = int(hashlib.md5(article['title'].encode('utf-8')).hexdigest(), 16) % 100
        if hash_mod_100 >= args.b_percentage:
            a_articles.append(article)
        else:
            b_articles.append(article)

    with open(args.output_a, 'w') as fileobj:
        fileobj.write(json.dumps(a_articles))
    with open(args.output_b, 'w') as fileobj:
        fileobj.write(json.dumps(b_articles))
