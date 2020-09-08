from d3m import container


class ContainersWithMetadata:
    params = [[True, False], [100, 300, 500, 700, 900]]
    param_names = ['compact', 'columns']

    def setup(self, compact, columns):
        # Compacting a DataFrame with more than 300 columns timeouts the benchmark and fails it.
        # By raising a "NotImplementedError exception such combinations are skipped.
        if compact and columns > 300:
            raise NotImplementedError

    def time_dataframe(self, compact, columns):
        df = container.DataFrame({str(i): [j for j in range(5)] for i in range(columns)}, columns=[str(i) for i in range(columns)], generate_metadata=False)
        df.metadata.generate(df, compact=compact)

    def time_columns(self, compact, columns):
        dfs = [
            container.DataFrame({str(i): [j for j in range(5, 10)]}, columns=[str(i)], generate_metadata=False)
            for i in range(int(columns / 2))
        ]
        for df in dfs:
            df.metadata.generate(df, compact=compact)


class ContainersWithoutMetadata:
    params = [[100, 300, 500, 700, 900]]
    param_names = ['columns']

    def time_dataframe(self, columns):
        container.DataFrame({str(i): [j for j in range(5)] for i in range(columns)}, columns=[str(i) for i in range(columns)], generate_metadata=False)

    def time_columns(self, columns):
        [
            container.DataFrame({str(i): [j for j in range(5, 10)]}, columns=[str(i)], generate_metadata=False)
            for i in range(int(columns / 2))
        ]
