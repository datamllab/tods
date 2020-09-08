from d3m import container
from d3m.base import utils as base_utils


class CombineColumns:
    params = [[100, 300, 500, 700, 900]]
    param_names = ['columns']

    def setup(self, columns):
        self.large_dataframe_with_many_columns = container.DataFrame({str(i): [j for j in range(5)] for i in range(columns)}, columns=[str(i) for i in range(columns)], generate_metadata=True)
        self.list_of_many_dataframe_columns = [
            container.DataFrame({str(i): [j for j in range(5, 10)]}, columns=[str(i)], generate_metadata=True)
            for i in range(int(columns / 2))
        ]

    def time_append(self, columns):
        base_utils.combine_columns(
            self.large_dataframe_with_many_columns,
            list(range(int(columns / 4), int(columns / 2))),  # Just 1/4 of columns.
            self.list_of_many_dataframe_columns,
            return_result='append',
            add_index_columns=True,
        )

    def time_replace(self, columns):
        base_utils.combine_columns(
            self.large_dataframe_with_many_columns,
            list(range(int(columns / 4), int(columns / 2))),  # Just 1/4 of columns.
            self.list_of_many_dataframe_columns,
            return_result='replace',
            add_index_columns=True,
        )

    def time_new(self, columns):
        base_utils.combine_columns(
            self.large_dataframe_with_many_columns,
            list(range(int(columns / 4), int(columns / 2))),  # Just 1/4 of columns.
            self.list_of_many_dataframe_columns,
            return_result='new',
            add_index_columns=True,
        )
