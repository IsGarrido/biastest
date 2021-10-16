from typing import List

from src.FillMask.HuggingFace.FillMaskTask import FillMaskTask
from src.FillMask.Vm.FillMaskModel import FillMaskModel
from src.FillMask.Vm.TaskResult import TaskResult
from src.Utils.FileWriterHelper import FileWriterHelper
from src.Utils.HuggingHelper import HuggingHelper
from src.Utils.FileHelperReader import FileHelperReader as Reader
from src.Utils.JsonHelper import JsonHelper


class Core:
    sentences: list[list[str]]
    uncased_sentences: list[list[str]]
    models: list[FillMaskModel]

    def __init__(self):
        self.sentences = Reader.read_tsv("~/data/FillMask/sentences.tsv")
        self.uncased_sentences = [[HuggingHelper.lower(p[0]), HuggingHelper.lower(p[1])] for p in self.sentences]
        self.types = ["male", "female"]

        model_names = Reader.read_tsv("~/data/FillMask/models.tsv")
        self.models = [FillMaskModel.from_tsv(model) for model in model_names ]

        self.run()

    def run(self):

        arr = []
        for model_item in self.models:
            model, tokenizer = model_item.get()
            task = FillMaskTask(model, tokenizer, 50)
            for sentence_pair in self.sentences:
                results = task.fill_sentences(sentence_pair, model_item.mask)
                sentence_results = TaskResult.assignTypes(self.types, results)
                arr.extend(sentence_results)

        json = JsonHelper.encode(arr)
        FileWriterHelper.write("~/result_data/FillMask/predictions.json", json)


Core()