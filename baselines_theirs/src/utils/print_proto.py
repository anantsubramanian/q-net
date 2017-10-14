import sys

from google.protobuf import reflection as _reflection

from proto import CoreNLP_pb2, dataset_pb2, training_dataset_pb2
from proto.io import ReadArticle

MODULES = [CoreNLP_pb2, dataset_pb2, training_dataset_pb2]

if __name__ == '__main__':
    msg_type = sys.argv[1]
    input_file = sys.argv[2]
    
    cls = None
    for module in MODULES:
        for var_name in dir(module):
            var = getattr(module, var_name)
            if isinstance(var, _reflection.GeneratedProtocolMessageType):
                if var.DESCRIPTOR.full_name == msg_type:
                    cls = var
    if cls is None:
        print >> sys.stderr, 'Unknown message type. Please enter the full name.'
        exit(1)

    with open(sys.argv[2], 'rb') as fileobj:
        while True:
            article = ReadArticle(fileobj, cls)
            if article is None:
                break
            article_str = str(article)
            article_str = ('{\n' + article_str).replace('\n', '\n  ').strip() + '\n}\n'
            print article_str
