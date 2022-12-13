import pandas as pd
from relhelperspy.primitives.dict_helper import DictHelper as _dict

class EvaluateCategoriesDataService():

    def add_is_adjective_column(self, df: pd.DataFrame):

        df['is_adjective'] = df["pos_tag"] == "AQ"
        df.drop(columns=['pos_tag'], inplace=True)

        return df

    def add_sentiment_columns(self, df: pd.DataFrame):
        df['sentiment_pos'] = df["sentiment"] == "POS"
        df['sentiment_neg'] = df["sentiment"] == "NEG"
        df.drop(columns=['sentiment'], inplace=True)

        return df
    
    def __aggregate(self, df_data, cols):
        base_df = df_data.groupby(cols, as_index=False).agg(
                score_sum=('score', 'sum'),

                score_min=('score', 'min'),
                score_max=('score', 'max'),
                score_mean=('score', 'mean'),

                count=('rsv', 'count'),

                adj_cnt=('is_adjective', 'sum'),
                snt_pos_cnt = ('sentiment_pos', 'sum'),
                snt_neg_cnt = ('sentiment_neg', 'sum')
            )
        return base_df
                
    def __calculate_rsv(self, df_data, cols):
        filtered_df = df_data[df_data["is_adjective"]];

        # If we are working with adjective, we only take RSV for those with adjectives
        if len(filtered_df) > 0:
            return df_data[df_data["is_adjective"]].groupby(cols, as_index=False).agg(
                rsv_sum=('rsv', 'sum'),

                rsv_min=('rsv', 'min'),
                rsv_max=('rsv', 'max'),
                rsv_mean=('rsv', 'mean'),
            )

        else:
            return df_data.groupby(cols, as_index=False).agg(
                rsv_sum=('rsv', 'sum'),

                rsv_min=('rsv', 'min'),
                rsv_max=('rsv', 'max'),
                rsv_mean=('rsv', 'mean'),
            )

    def __merge(self, df1, df2, cols):
        merged_df = df1.merge(df2, how='outer', on = cols)
        return merged_df

    
    def add_category_column(self, df: pd.DataFrame, categories: 'dict[str, str]'):
        category_lookup = _dict.as_lookup(categories)
        df["category"] = df.apply(
            lambda row: category_lookup.get(row['word'], 'unknown')
        , axis=1 ) 
        return df

    def add_adjective_proportion(self, df: pd.DataFrame) -> pd.DataFrame:
        df["adj_prop"] = (df["adj_cnt"] / df["count"]) * 100
        return df

    def group_by_sentence_fn(self, df_data: pd.DataFrame) -> pd.DataFrame:
        
        grouping_columns = ['dimension', 'model', 'category', 'sentence']
        base_df = self.__aggregate(df_data, grouping_columns)
        rsv_df = self.__calculate_rsv(df_data, grouping_columns)

        return self.__merge(base_df, rsv_df, grouping_columns )
        # return grouped_res[grouped_res.category != 'unknown']


    def group_by_category_fn(self, df_data: pd.DataFrame) -> pd.DataFrame:

        grouping_columns = ['dimension', 'model', 'category']
        base_df = self.__aggregate(df_data, grouping_columns)
        rsv_df = self.__calculate_rsv(df_data, grouping_columns)

        return self.__merge(base_df, rsv_df, grouping_columns )


    def group_by_dimension_fn(self, df_data: pd.DataFrame) -> pd.DataFrame:

        grouping_columns = ['dimension', 'model']
        base_df = self.__aggregate(df_data, grouping_columns)
        rsv_df = self.__calculate_rsv(df_data, grouping_columns)

        return self.__merge(base_df, rsv_df, grouping_columns )


    def group_by_model_fn(self, df_data: pd.DataFrame) -> pd.DataFrame:

        grouping_columns = ['model']
        base_df = self.__aggregate(df_data, grouping_columns)
        rsv_df = self.__calculate_rsv(df_data, grouping_columns)

        return self.__merge(base_df, rsv_df, grouping_columns )


    def group_sentences(self, df_data):

        grouping_columns = ['sentence']
        base_df = self.__aggregate(df_data, grouping_columns)
        rsv_df = self.__calculate_rsv(df_data, grouping_columns)

        return self.__merge(base_df, rsv_df, grouping_columns )


    def group_sentences_with_dimensions(self, df_data):

        grouping_columns = ['sentence', 'dimension']
        base_df = self.__aggregate(df_data, grouping_columns)
        rsv_df = self.__calculate_rsv(df_data, grouping_columns)

        return self.__merge(base_df, rsv_df, grouping_columns )

    def group_model_category(self, df_data):

        grouping_columns = ['model', 'category']
        base_df = self.__aggregate(df_data, grouping_columns)
        rsv_df = self.__calculate_rsv(df_data, grouping_columns)

        merged_df = self.__merge(base_df, rsv_df, grouping_columns )

        # https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby/57359372#57359372
        # Ex 25% male, 75% female
        subgrouping_columns = ['model']
        merged_df['score_prop'] = 100 * merged_df['score_sum'] / merged_df.groupby(subgrouping_columns)['score_sum'].transform('sum')
        merged_df['rsv_prop'] = 100 * merged_df['rsv_sum'] / merged_df.groupby(subgrouping_columns)['rsv_sum'].transform('sum')

        # https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby/57359372#57359372
        # Ex male->body 10%, female->body 14%
        overall_subgrouping_columns = ['model']
        merged_df['score_prop_overall'] = 100 * merged_df['score_sum'] / merged_df.groupby(overall_subgrouping_columns)['score_sum'].transform('sum')
        merged_df['rsv_prop_overall'] = 100 * merged_df['rsv_sum'] / merged_df.groupby(overall_subgrouping_columns)['rsv_sum'].transform('sum')

        return merged_df


    def group_model_category_with_dimension(self, df_data):

        grouping_columns = ['model', 'category', 'dimension']
        base_df = self.__aggregate(df_data, grouping_columns)
        rsv_df = self.__calculate_rsv(df_data, grouping_columns)

        merged_df = self.__merge(base_df, rsv_df, grouping_columns )

        # https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby/57359372#57359372
        # Ex 25% male, 75% female
        subgrouping_columns = ['model', 'category']
        merged_df['score_prop'] = 100 * merged_df['score_sum'] / merged_df.groupby(subgrouping_columns)['score_sum'].transform('sum')
        merged_df['rsv_prop'] = 100 * merged_df['rsv_sum'] / merged_df.groupby(subgrouping_columns)['rsv_sum'].transform('sum')

        # https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby/57359372#57359372
        # Ex male->body 10%, female->body 14%
        overall_subgrouping_columns = ['model', 'dimension']
        merged_df['score_prop_overall'] = 100 * merged_df['score_sum'] / merged_df.groupby(overall_subgrouping_columns)['score_sum'].transform('sum')
        merged_df['rsv_prop_overall'] = 100 * merged_df['rsv_sum'] / merged_df.groupby(overall_subgrouping_columns)['rsv_sum'].transform('sum')

        return merged_df