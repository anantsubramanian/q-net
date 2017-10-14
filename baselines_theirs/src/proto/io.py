from google.protobuf.internal.decoder import _DecodeVarint32
from google.protobuf.internal.encoder import _EncodeVarint
import google.protobuf.reflection

from proto import CoreNLP_pb2
from proto import dataset_pb2
from proto import training_dataset_pb2


def WriteArticle(article, fileobj):
    """
    Writes the given Article protocol buffer to the given file-like object.
    """
    msg = article.SerializeToString()
    _EncodeVarint(fileobj.write, len(msg))
    fileobj.write(msg)


def ReadArticle(fileobj, cls=dataset_pb2.Article):
    """
    Reads a single Article protocol buffer from the given file-like object.
    """
    hdr = fileobj.read(4)
    if len(hdr) == 0:
        return None
    msg_length, hdr_length = _DecodeVarint32(hdr, 0)
    msg = hdr[hdr_length:] + fileobj.read(msg_length - (4 - hdr_length))

    article = cls()
    article.ParseFromString(msg)
    return article


def ReadArticles(filename, n=None, cls=dataset_pb2.Article):
    """
    Reads all articles from the file with the given name.
    """
    articles = []
    with open(filename, 'rb') as f:
        while True:
            article = ReadArticle(f, cls)
            if article is None:
                return articles
            articles.append(article)

            if n is not None and len(articles) == n:
                return articles


# Hack to get pickling to work.
def fix_module(module):
    for el in module.__dict__.itervalues():
        if isinstance(el, google.protobuf.reflection.GeneratedProtocolMessageType):
            el.__module__ = module.__name__
fix_module(CoreNLP_pb2)
fix_module(dataset_pb2)
fix_module(training_dataset_pb2)
