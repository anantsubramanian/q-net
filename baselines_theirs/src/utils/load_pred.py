import json

with open("/Users/Jian/Downloads/eval_3000.eval", "r") as fp:
    prediction = json.load(fp)
    trainPred = prediction["train-pred"]
    evalPred = prediction["eval-pred"]

with open("/Users/Jian/Downloads/train-pred.pred", "w") as fp:
    json.dump(trainPred, fp)
with open("/Users/Jian/Downloads/eval-pred.pred", "w") as fp:
    json.dump(evalPred, fp)
