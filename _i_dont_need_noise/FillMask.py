from dataclass.filltemplate.fill_template_config import FillTemplateConfig
from source.FillMaskUtils.GroupedFillMask import GroupedFillMask
from source.FillMaskUtils.RunResult import RunResult
from source.FillMaskUtils.CategorizacionConfig import CategorizacionConfig

# Helpers
import relhelpers.io.read_helper as _read
import relhelpers.io.write_helper as _write
import relhelpers.date.date_helper as _date
import relhelpers.primitives.string_helper as _string
import relhelpers.primitives.list_helper as _list
import relhelpers.huggingface.model_helper as _hf_model
import relhelpers.stats.statistical_analysis_helper as _stats

print( 1 + "")


cat_config_ismael = CategorizacionConfig(
    "result_fillmask/categorias_ismael",
    "./data/CategoriasAdjetivos/excel_ismael_v2.tsv",
    "./data/FillMask/sentences.tsv",
    True
)

cat_config_polaridad_visibilidad = CategorizacionConfig(
    "result_fillmask/categorias_polaridad_visibilidad",
    "./data/CategoriasAdjetivos/polaridad_visibilidad.tsv",
    "./data/FillMask/sentences.tsv",
    True
)

cat_config_polaridad_visibilidad_negadas = CategorizacionConfig(
    "result_fillmask/categorias_polaridad_visibilidad_negadas",
    "./data/CategoriasAdjetivos/polaridad_visibilidad.tsv",
    "./data/FillMask/sentences_neg.tsv",
    True
)

cat_config_polaridad_foa_foa = CategorizacionConfig(
    "result_fillmask/categorias_polaridad_foa_foa",
    "./data/CategoriasAdjetivos/polaridad_foa_foa.tsv",
    "./data/FillMask/sentences.tsv",
    True
)

cat_config_polaridad_foa_foa_with_visibles = CategorizacionConfig(
    "result_fillmask/categorias_polaridad_foa_foa_with_visibles",
    "./data/CategoriasAdjetivos/polaridad_foa_foa_with_visibles.tsv",
    "./data/FillMask/sentences.tsv",
    True
)

cat_config_yulia = CategorizacionConfig(
    "result_fillmask/categorias_yulia",
    "./data/CategoriasAdjetivos/yulia.tsv",
    "./data/FillMask/sentences.tsv",
    False
)

'''
cat_config_profesiones =  CategorizacionConfig(
    "result_fillmask_profesiones/base",
    "./data/CategoriasAdjetivos/profesiones_10_cnae_2021t2.tsv",
    "./data/FillMask/sentences_profesiones.tsv",
    False,
    10
)
'''

'''
cconfig = cat_config_yulia
cconfig = cat_config_polaridad_foa_foa
cconfig = cat_config_ismael
cconfig = cat_config_polaridad_visibilidad
'''
cconfig = cat_config_polaridad_foa_foa

# constantes
T = "\t"
#RESULT_PATH = "result_fillmask/categorias_ismael"
TOTAL_CAT = "[_TOTAL]"
UNKOWN_CAT = "[???]"
WORD_MIN_LEN = 4
WRITE_DEBUG = False
WRITE_STATS = False

# Inicializar cosas en espacio común, cuando esté estado hay que darle una vuelta, muy cutre esto

# Stores
adjectives_map = _read.read_lines_as_dict("../TextTools/GenerarListadoPalabras/result/adjetivos.txt")

if cconfig.categories_ready:
    adjetivos_categorizados = _read.read_lines_as_col_excel_asdict(cconfig.categories_source_file)
    adjetivos_categorias = _list.unique(list(adjetivos_categorizados.values())) # Bastante bruto esto
else:
    adjetivos_categorizados = {}
    adjetivos_categorias = []

# Comunes a todas las runs
all_filling_words = []
all_filling_adjectives = []
run_id = 0

run_results = []

'''
generator = pipeline("text-generation", model = "dccuchile/bert-base-spanish-wwm-uncased", tokenizer= "dccuchile/bert-base-spanish-wwm-uncased")

generator(
    "El es muy",
    max_length = 30,
    num_return_sentences = 30,
)
'''

def run_grouped(model, modelname, tokenizer, sentences):
    filler = GroupedFillMask(model, modelname, tokenizer, cconfig.RESULT_PATH, cconfig.quantity, WRITE_DEBUG).run_for_sentences(sentences)
    return filler


def save_run(model_name, retrieval_status_values, probabilities, kind="m"):

    # ReFormatear resultados

    l = []
    l_adj = []
    category_count = {}
    category_retrieval_status_values = {}
    category_probability = {}

    # Inicializar el mapa
    for category in adjetivos_categorias:
        category_count[category] = 0
        category_retrieval_status_values[category] = 0
        category_probability[category] = 0

    category_count[UNKOWN_CAT] = 0
    category_retrieval_status_values[UNKOWN_CAT] = 0
    category_probability[UNKOWN_CAT] = 0

    category_retrieval_status_values[TOTAL_CAT] = 0
    category_count[TOTAL_CAT] = 0
    category_probability[TOTAL_CAT] = 0

    for key, retrieval_status_values_value in retrieval_status_values.items():
        probability_value = probabilities[key]

        l.append(_string.from_int(retrieval_status_values_value, 4) + T + key)
        all_filling_words.append(key)

        # Solo si es un adjetivo OK
        if key in adjectives_map or not cconfig.check_is_adjective:
            l_adj.append(_string.from_int(retrieval_status_values_value, 4) + T + key)
            all_filling_adjectives.append(key)

            # Buscar la categoria OK
            category = UNKOWN_CAT
            if key in adjetivos_categorizados:
                category = adjetivos_categorizados[key]

            # Agregar valores
            category_count[category] = category_count[category] + 1
            category_retrieval_status_values[category] = category_retrieval_status_values[category] + retrieval_status_values_value
            category_probability[category] = category_probability[category] + probability_value

            # Agregar al total
            category_count[TOTAL_CAT] = category_count[TOTAL_CAT] + 1
            category_retrieval_status_values[TOTAL_CAT] = category_retrieval_status_values[TOTAL_CAT] + retrieval_status_values_value
            category_probability[TOTAL_CAT] = category_probability[TOTAL_CAT] + probability_value

    l_category = []
    dict_results = {}
    for category_key in category_count.keys():

        # Count
        count = category_count[category_key]
        prc_count = (count*100) / category_count[TOTAL_CAT]

        # retrieval_status_values
        retrieval_status_values = category_retrieval_status_values[category_key]
        total_retrieval_status_values = category_retrieval_status_values[TOTAL_CAT]
        prc_retrieval_status_values = (retrieval_status_values * 100) / total_retrieval_status_values

        # probabilities
        probability = category_probability[category_key]
        total_probability = category_probability[TOTAL_CAT]
        prc_probability = (probability * 100) / total_probability

        rresult = RunResult(category_key, count, prc_count, retrieval_status_values, prc_retrieval_status_values, probability, prc_probability)
        dict_results[category_key] = rresult

        l_category.append(category_key + T + str(count) + T + str( prc_count ) + T + str(retrieval_status_values) + T + str( prc_retrieval_status_values ) + T + str(probability) + T + str(prc_probability))

    # Ordenar ahora que son texto
    l.sort(reverse=True)
    l_adj.sort(reverse=True)
    l_category.sort(reverse=True)

    # Añadir cabeceras
    l_category.insert(0, "[CAT]" + T + "Count" + T + "PRC_Count" + T + "retrieval_status_values" + T + "PRC_retrieval_status_values" + T + "probability" + T + "PRC_probability")

    # Juntar lineas
    data = "\n".join(l)
    data_adj = "\n".join(l_adj) + "\n" + _date.FechaHoraTextual()
    data_category = "\n".join(l_category)

    # Pasar a disco
    path = cconfig.RESULT_PATH + "/run_" + str(run_id) + "_" + kind + "_" + _string.as_file_name(model_name)

    if WRITE_DEBUG:
        _write.txt(data, path + ".csv")
        _write.txt(data_adj, path + "_adj.csv")
        _write.txt(data_category, path + "_cat.csv")

    return dict_results

def run_global_stats():
    #run_results
    # adjetivos_categorias

    models =    [x[0] for x in run_results]
    m_result_tables = [x[1] for x in run_results]
    f_result_tables = [x[2] for x in run_results]
    attrs = ['count', 'prc_count', 'retrieval_status_values', 'prc_retrieval_status_values']

    # Quiero buscar correlaciones para TODOS los attributos njumericos
    for attr in attrs:

        # La correlación se busca para cada categoría
        for cat in adjetivos_categorias:
            l_before = []
            l_after = []
            l_model = []

            # Para cada par de tablas de resultados, buscamos correlaciones para el attributo ATTR
            for idx, val in enumerate(m_result_tables):

                m_result_table = m_result_tables[idx]
                f_result_table = f_result_tables[idx]
                l_model.append(models[idx])

                m_result_cat = m_result_table[cat]
                f_result_cat = f_result_table[cat]

                val_m = getattr(m_result_cat, attr)
                val_f = getattr(f_result_cat, attr)

                l_before.append(val_m)
                l_after.append(val_f)

            result_text = _stats.run_tests_labeled(l_before, l_after)

            posfix = _string.as_file_name(cat)  + "_" + str(attr)

            # Escribir listas para posterior revisión
            data_m = _list.list_as_file(_list.list_as_str_list(l_before))
            data_f = _list.list_as_file(_list.list_as_str_list(l_after))

            _write.txt(data_m, cconfig.RESULT_PATH + "/stats_source_" + posfix + "_m.txt")
            _write.txt(data_f, cconfig.RESULT_PATH + "/stats_source_" + posfix + "_f.txt")

            l_both = []
            for idx, val in enumerate(l_before):
                m_val = l_before[idx]
                f_val = l_after[idx]
                modelname = l_model[idx]
                arrow = ">" if m_val > f_val else "<"
                l_both.append( str(m_val) + T + arrow + T + str(f_val) + T + modelname)

            str_both_l = _list.list_as_str_list(l_both)
            str_both_l.insert(0, "[MASC]" + T + " " + T + "[FEM]" + T + "[model_name]")
            str_both_l.insert(0, "")
            str_both_l.insert(0, "")
            str_both_l.insert(0, _string.as_file_name(cat) + "," + str(attr))

            data_both = _list.list_as_file(str_both_l, False)

            _write.txt(data_both, cconfig.RESULT_PATH + "/stats_both_" + posfix + ".csv")

            # Escribir resultado
            path = cconfig.RESULT_PATH + "/stats_result_" + posfix + ".txt"
            _write.txt(result_text, path )

def run(modelname, tokenizername, MASK, sentences):

    print("Loading model " + modelname + " with mask " + MASK)
    model, tokenizer = _hf_model.load_model(modelname, tokenizername)
    print("Model loaded")

    sentences_m = [sentence[0].replace("[MASK]", MASK) for sentence in sentences]
    sentences_f = [sentence[1].replace("[MASK]", MASK) for sentence in sentences]

    c_m, retrieval_status_values_m, probability_m = run_grouped(model, modelname, tokenizer, sentences_m)
    c_f, retrieval_status_values_f, probability_f = run_grouped(model, modelname, tokenizer, sentences_f)

    result_table_m = save_run(modelname, retrieval_status_values_m, probability_m, "m")
    result_table_f = save_run(modelname, retrieval_status_values_f, probability_f, "f")
    run_results.append((modelname, result_table_m, result_table_f))

    print("OK => " + modelname)

sentences = _read.read_paired_tsv(cconfig.sentences_path)

uncased_sentences = [ [p[0].lower().replace("[mask]", "[MASK]"), p[1].lower().replace("[mask]", "[MASK]")] for p in sentences]
models = _read.read_paired_tsv("./data/FillMask/models.tsv")

for idx, model in enumerate(models):
    run_id = model[0]
    sentence_list = sentences if model[4] == "cased" else uncased_sentences
    run(model[1], model[2], model[3], sentence_list)
    print("Finalizado modelo nro " + str(idx))

data = _list.list_as_file(all_filling_words)
_write.txt(data, cconfig.RESULT_PATH + "/summary_all_filling_words.csv")

data = _list.list_as_file(all_filling_adjectives)
_write.txt(data, cconfig.RESULT_PATH + "/summary_all_filling_adjectives.csv")


if cconfig.categories_ready:

    if WRITE_STATS:
        run_global_stats()

    adjetivos_sin_categorizar = filter( lambda adjetivo: not adjetivo in adjetivos_categorizados, all_filling_adjectives)
    data = _list.list_as_file(adjetivos_sin_categorizar)
    _write.txt(data, cconfig.RESULT_PATH + "/summary_adj_missing_category.csv")
    _write.json(run_results, cconfig.RESULT_PATH + "/run_result.json")

