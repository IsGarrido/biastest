

import pandas as pd
from transformers import FillMaskPipeline
from dataclass.fill_template_config import FillTemplateConfig
from dataclass.model_config import ModelConfig
from relhelpers.io.project_helper import ProjectHelper as _project
from relhelpers.huggingface.model_helper import HuggingFaceModelHelper as _hf_model
from relhelpers.huggingface.fillmask_helper import FillMaskHelper as _hf_fillmask
from relhelpers.pandas.pandas_helper import PandasHelper as _pd
from relhelpers.primitives.string_helper import StringHelper as _string
from relhelpers.io.write_helper import WriteHelper as _write

import warnings
warnings.filterwarnings("ignore")

class FillTemplate:

    def __init__(self, cfg: FillTemplateConfig) -> None:

        self.cfg = cfg        
        self.data = pd.DataFrame()
        self.model_data = pd.DataFrame()

        folder = _project.result_path(self.__class__.__name__, None)
        _write.create_dir(folder)

        self.run()

    def run(self):
        # TODO: Generalizar y cachear
        cased_templates_df = _pd.read_tsv(self.cfg.templates_path)
        uncased_templates_df = _pd.apply_all_cells(cased_templates_df, _hf_fillmask.to_lower_but_mask)

        cased_templates_roberta_df = _pd.apply_all_cells(cased_templates_df, _hf_fillmask.to_robert_mask)
        uncased_templates_roberta_df = _pd.apply_all_cells(uncased_templates_df, _hf_fillmask.to_robert_mask)

        models_df = _pd.read_tsv(_project.data_path("FillMask", "models.tsv"))
        models: 'list[ModelConfig]' = models_df.apply( lambda row: ModelConfig(row[0],row[1],row[2],row[3], row[3] == 'cased'), axis = 1)

        for model in models:

            # TODO: Generalizar y cachear
            if model.cased:
                if model.mask == "[MASK]":
                    templates = cased_templates_df
                else:
                    templates = cased_templates_roberta_df
            else:
                if model.mask == "[MASK]":
                    templates = uncased_templates_df
                else:
                    templates = uncased_templates_roberta_df

            self.run_for_model(model.name, model.tokenizer, model.mask, templates)
        
        self.export_all_results()

    def run_for_model(self, model_name: str, tokenizer_name: str, mask: str, templates_df: pd.DataFrame):

        model, tokenizer = _hf_model.load_model(model_name, tokenizer_name)
        pipeline = FillMaskPipeline(model, tokenizer, top_k=self.cfg.n_predictions, device=model.device.index)

        dimensions = templates_df.columns
        for dimension in dimensions:
            dimension_df = templates_df[dimension]
            self.run_for_dimension(pipeline, model_name, dimension_df, dimension)
        
        self.data = self.data.append(self.model_data)
        self.export_model_result(model_name)
        self.model_data = pd.DataFrame()

        # c_m, retrieval_status_values_m, probability_m = run_grouped(model, model_name, tokenizer, sentences_m)
        # c_f, retrieval_status_values_f, probability_f = run_grouped(model, modelname, tokenizer, sentences_f)
    def run_for_dimension(self, pipeline: FillMaskPipeline, model_name: str, df: pd.DataFrame, dimension: str):
        [self.run_for_sentence(pipeline, model_name, sentence, dimension) for sentence in df]

    def run_for_sentence(self, pipeline: FillMaskPipeline,  model_name: str, sentence: str, dimension: str):
        # Predict
        res = pipeline(sentence)

        # To pandas
        res_df = _pd.from_dict(res)

        # Delete extra
        res_df = _pd.remove_col(res_df, 'sequence')

        # Add context
        res_df['sentence'] = sentence   # He is [MASK]
        res_df['model'] = model_name    # beto
        res_df['dimension'] = dimension # m/f

        # Alter
        res_df.reset_index(inplace=True)
        # res_df['idx'] = res_df.index
        res_df['rsv'] = [len(res_df)]*len(res_df) - res_df.index 

        self.model_data = self.model_data.append(res_df) 

    def export_model_result(self, model_name):
        path = _project.result_path(self.__class__.__name__, _string.as_file_name(model_name) + ".tsv" )
        _pd.save(self.model_data, path)

    def export_all_results(self):
        path = _project.result_path(self.__class__.__name__, self.__class__.__name__ + ".tsv" )
        _pd.save(self.data, path)

        # result_table_m = save_run(modelname, retrieval_status_values_m, probability_m, "m")
        # result_table_f = save_run(modelname, retrieval_status_values_f, probability_f, "f")
        # run_results.append((modelname, result_table_m, result_table_f))

        # print("OK => " + modelname)


cfg = FillTemplateConfig(
        'genre',
        'Spanish Genre',
        _project.data_path('FillMask','sentences.tsv')
)
print(cfg)

FillTemplate(cfg)


'''

class GroupedFillMask:

    def run_for_sentences(self, sentences):
        for sentence in sentences:
            self.run_for_text(sentence)

        counts = dict(sorted(self.grouped_count.items(), key=lambda item: item[1], reverse=True))
        retrieval_status_values = dict(sorted(self.grouped_retrieval_status_values.items(), key=lambda item: item[1], reverse=True))
        #probabilities = { key: sum(value)/len(value) for key,value in self.grouped_model_probabilities.items()}
        probabilities = { key: sum(value) for key,value in self.grouped_model_probabilities.items()}

        return counts, retrieval_status_values, probabilities

    def run_for_text(self, line):

        if line == "él ha conseguido el trabajo ya que es muy [MASK]." or line == "ella ha conseguido el trabajo ya que es muy [MASK].":
            print(1)

        res = self.pipeline(line)
        self.process_result(res, line)


    def valid_token(self, token: str):

        clean = token.replace("[", "").replace("]", "").replace(".", "")

        if len(clean) < WORD_MIN_LEN:
            return False
        if "#" in token:
            return False

        return True

    def save_stats(self, word: str, probability, idx):

        word = word.strip()
        if not self.valid_token(word):
            return

        # Count
        if word in self.grouped_count:
            self.grouped_count[word] = self.grouped_count[word] + 1
        else:
            self.grouped_count[word] = 1

        retrieval_status_values = self.result_qty - idx

        if word in self.grouped_retrieval_status_values:
            self.grouped_retrieval_status_values[word] = self.grouped_retrieval_status_values[word] + retrieval_status_values
        else:
            self.grouped_retrieval_status_values[word] = retrieval_status_values

        if word in self.grouped_model_probabilities:
            self.grouped_model_probabilities[word].append(probability)
        else:
            self.grouped_model_probabilities[word] = []
            self.grouped_model_probabilities[word].append(probability)


    def process_result(self, items, orig_line):
        l = []
        for idx, item in enumerate(items):
            word: str = item["token_str"].lower()
            probability = item["score"]
            self.save_stats(word, probability, idx)

            line = str(idx) + T + word + T + str(item["token"])
            l.append(line)

        if self.write_files:
            text = "\n".join(l)
            path = self.result_path + "/" + _string.as_file_name(self.modelname) + "/" + _string.as_file_name(orig_line) + ".csv"
            _write.txt(text, path)

'''