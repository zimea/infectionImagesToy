from asyncio.log import logger
import os
from xmlrpc.client import boolean
import pandas as pd
import numpy as np
import glob
from Utility import timeit
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr


class MorpheusReader:
    def __init__(self, config, prior_means, prior_stds):
        self.config = config
        self.prior_means = prior_means
        self.prior_stds = prior_stds

    def calculate_V(self, path: str):
        df = pd.read_csv(path, sep="\t")
        df = df[df[str(self.config.grid_size)] != self.config.grid_size]
        morpheus_ts = np.repeat(
            range(0, self.config.timesteps + 1), self.config.grid_size
        )
        times = np.tile(morpheus_ts, int(len(df.index) / morpheus_ts.shape[0]))
        if len(times) != len(df.index):
            print(path)
        df["time"] = np.tile(morpheus_ts, int(len(df.index) / morpheus_ts.shape[0]))
        df["sum_V"] = df.iloc[:, 1 : self.config.grid_size + 1].sum(axis=1)
        df = df.drop(columns=[str(x) for x in range(0, self.config.grid_size + 1)])
        df = df.groupby(list(df.columns[:-1])).agg({"sum_V": "sum"}).reset_index()
        return np.expand_dims(df["sum_V"], axis=1)[
            self.config.cut_off_start
            + 1 : self.config.timesteps
            - self.config.cut_off_end
        ]

    def calculate_volume(self, path: str):
        logger_state = "logger_1.csv"
        logger_volume = "logger_4_cell.id.csv"
        state = pd.read_csv(os.path.join(path, logger_state), sep="\t").rename(
            columns={"cell.id": "id"}
        )
        volume = np.genfromtxt(os.path.join(path, logger_volume), delimiter="\t")
        lines_per_timepoint = (self.config.timesteps + 1) * (self.config.grid_size + 1)
        volume = np.delete(
            volume,
            list(range(0, int(lines_per_timepoint), self.config.grid_size + 1)),
            axis=0,
        )

        first_ts_after_freeze = self.config.grid_size * (self.config.cut_off_start + 1)
        cell_id, counts = np.array(
            np.unique(
                np.array(volume)[
                    first_ts_after_freeze : first_ts_after_freeze
                    + self.config.grid_size,
                    1:,
                ],
                return_counts=True,
            )
        )
        counts = np.append(
            np.expand_dims(cell_id, axis=1), np.expand_dims(counts, axis=1), axis=1
        )
        if counts[0, 0] != 0:
            counts = np.insert(counts, 0, [[0, 0]], axis=0)
        counts = pd.DataFrame(counts, columns=["id", "count"])
        df = pd.merge(left=state, right=counts, how="left", on=["id"])
        df = df.groupby(["time", "V"]).agg({"count": "sum"}).reset_index()
        return np.expand_dims(df.query("V == 1")["count"], axis=1)

    @timeit
    def read_offline_data_1d(self, path: str, workdir):
        with open(os.path.join(workdir, "log_read_offline.txt"), "w") as logfile:
            with redirect_stdout(logfile), redirect_stderr(logfile):
                path_list = glob.glob(path)
                nr_of_params = self.config.param_nr

                n_sim = len(path_list)
                dfs = np.empty(
                    (
                        n_sim,
                        (
                            self.config.timesteps
                            - 1
                            - self.config.cut_off_start
                            - self.config.cut_off_end
                        ),
                        3,
                    ),
                    dtype=np.float32,
                )
                params = np.empty((n_sim, nr_of_params), dtype=np.float32)
                invalidIndices = []

                sum_rejected = 0
                for path in tqdm(range(n_sim)):
                    pathname = path_list[path]
                    print(pathname)
                    filename = os.path.join(pathname, "logger_2.csv")
                    filename_V = os.path.join(pathname, "logger_6_Ve.csv")
                    df = pd.read_csv(filename, index_col=None, header=0, delimiter="\t")
                    path_split = filename.split("/")[len(filename.split("/")) - 2]
                    if "e" in path_split:
                        invalidIndices.append(path)
                        continue
                    if path_split.startswith("sweep") or path_split.startswith("DV"):
                        start_nr = 1
                    else:
                        start_nr = 0
                    params_split = path_split.split("_")[start_nr : nr_of_params + 1]
                    param_file = list(
                        map(lambda x: round(float(x.split("-")[1]), 3), params_split)
                    )
                    if self.config.rejectionBasedOnPrior(param_file) == True:
                        sum_rejected += 1
                        invalidIndices.append(path)
                        continue

                    df_tar = df["celltype.target.size"].values[:, np.newaxis][
                        self.config.cut_off_start
                        + 1 : self.config.timesteps
                        - self.config.cut_off_end
                    ]
                    df_inf = df["celltype.infected.size"].values[:, np.newaxis][
                        self.config.cut_off_start
                        + 1 : self.config.timesteps
                        - self.config.cut_off_end
                    ]
                    df_cells = np.append(df_tar, df_inf, axis=1)
                    try:
                        df_V = self.calculate_V(filename_V)
                        # I_volume = self.calculate_volume(pathname)[
                        #     self.config.cut_off_start
                        #     + 1 : self.config.timesteps
                        #     - self.config.cut_off_end
                        # ]
                    except Exception as error:
                        invalidIndices.append(path)
                        continue

                    sim = np.append(df_cells, df_V, axis=1)
                    if self.config.rejectionBasedOnSimulation(sim):
                        sum_rejected += 1
                        invalidIndices.append(path)
                        continue

                    if (
                        np.any(df_inf < 1)
                        or np.any(np.asarray(param_file) > 1)
                        or len(df_inf)
                        != (
                            self.config.timesteps
                            - 1
                            - self.config.cut_off_start
                            - self.config.cut_off_end
                        )
                    ):
                        invalidIndices.append(path)
                        continue
                    params[path] = param_file
                    # dfs[path] = np.append(
                    #     np.append(df_cells, df_V, axis=1), I_volume, axis=1
                    # )
                    dfs[path] = sim

                dfs = np.delete(dfs, invalidIndices, axis=0)
                params = np.delete(params, invalidIndices, axis=0)
                print("Read data in the form of: ", dfs.shape, "\n")
                print("Sum rejected: ", sum_rejected)

                return dfs, params

    def prepare_input(self, forward_dict):
        """Function to self.configure the simulated quantities (i.e., simulator outputs)
        into a neural network-friendly (BayesFlow) format.
        """

        # Prepare placeholder dict
        out_dict = {}

        # Convert data to logscale

        logdata = np.log1p(forward_dict["sim_data"]).astype(np.float64)

        # Extract prior draws and z-standardize with previously computed means
        params = forward_dict["prior_draws"].astype(np.float64)
        params = (params - self.prior_means) / self.prior_stds

        # Remove a batch if it contains nan, inf or -inf
        # idx_keep = np.all(np.isfinite(logdata), axis=(1, 2))

        # Add to keys
        out_dict["summary_conditions"] = logdata
        out_dict["parameters"] = params
        return out_dict

    def read_offline_split(self, path: str, workdir):
        with open(os.path.join(workdir, "log_read_offline.txt"), "w") as logfile:
            with redirect_stdout(logfile), redirect_stderr(logfile):
                path_to_split_data = os.path.join(
                    self.config.processed_data_path,
                    str(self.config.cell_nr - 1)
                    + "_"
                    + self.config.folder
                    + "_"
                    + self.config.rejectionBasedOnPrior.__name__
                    + "_"
                    + self.config.rejectionBasedOnSimulation.__name__,
                )
                files_exist = os.path.exists(
                    os.path.join(path_to_split_data, "train.npy")
                )
                reread_data = self.config.reread_data or not files_exist

                if reread_data == False:
                    train: dict = np.load(
                        os.path.join(path_to_split_data, "train.npy"), allow_pickle=True
                    )[()]
                    test: dict = np.load(
                        os.path.join(path_to_split_data, "test.npy"), allow_pickle=True
                    )[()]
                    validation: dict = np.load(
                        os.path.join(path_to_split_data, "validation.npy"),
                        allow_pickle=True,
                    )[()]
                else:
                    if not os.path.exists(path_to_split_data):
                        os.mkdir(path_to_split_data)

                    if self.config.image_data:
                        dfs, params = self.read_offline_data_2d(path, workdir)
                    else:
                        dfs, params = self.read_offline_data_1d(path, workdir)

                    indices = np.random.permutation(dfs.shape[0])
                    split_test, split_validation = (
                        int(dfs.shape[0] * self.config.test_ratio),
                        int(
                            dfs.shape[0]
                            * (self.config.validation_ratio + self.config.test_ratio)
                        ),
                    )
                    test_idx, validation_idx, training_idx = (
                        indices[:split_test],
                        indices[split_test:split_validation],
                        indices[split_validation:],
                    )
                    print("train.shape: ", training_idx.shape, "\n")
                    print("test.shape: ", test_idx.shape, "\n")
                    print("validation.shape: ", validation_idx.shape, "\n")
                    train = {
                        "sim_data": dfs[training_idx],
                        "prior_draws": params[training_idx],
                    }
                    test = {"sim_data": dfs[test_idx], "prior_draws": params[test_idx]}
                    validation = {
                        "sim_data": dfs[validation_idx],
                        "prior_draws": params[validation_idx],
                    }
                    np.save(
                        os.path.join(path_to_split_data, "train.npy"),
                        train,
                        allow_pickle=True,
                    )
                    np.save(
                        os.path.join(path_to_split_data, "test.npy"),
                        test,
                        allow_pickle=True,
                    )
                    np.save(
                        os.path.join(path_to_split_data, "validation.npy"),
                        validation,
                        allow_pickle=True,
                    )

                return train, test, validation

    def __read_2d_logger__(self, path: str, logger_name: str):
        logger = np.genfromtxt(os.path.join(path, logger_name), delimiter="\t")
        grid_size = logger[0, 0]
        logger = np.delete(
            logger, list(range(0, logger.shape[0], int(grid_size + 1))), axis=0
        )
        logger = logger[:, 1:]
        logger = logger.reshape((51, 45, 45))
        logger = logger[
            self.config.cut_off_start
            + 1 : self.config.timesteps
            - self.config.cut_off_end
        ]
        return logger

    def __read_1d_logger__(self, path: str, logger_name: str):
        logger = np.genfromtxt(os.path.join(path, logger_name), delimiter="\t")
        logger = logger[1:, :]
        logger = logger.reshape(
            (
                self.config.timesteps + 1,
                int(logger.shape[0] / (self.config.timesteps + 1)),
                logger.shape[1],
            )
        )
        logger = logger[
            self.config.cut_off_start
            + 1 : self.config.timesteps
            - self.config.cut_off_end
        ]
        return logger

    def __read_V_2d__(self, path: str):
        logger_name = "logger_6_Ve.csv"
        v_ext = self.__read_2d_logger__(path, logger_name)
        return v_ext

    def __read_I_2d__(self, path: str):
        logger_cell_id = "logger_4_cell.id.csv"
        logger_map_id_state = "logger_1.csv"
        cell_id = self.__read_2d_logger__(path, logger_cell_id)
        map_id_state = self.__read_1d_logger__(path, logger_map_id_state)

        map = np.zeros((map_id_state.shape[0], self.config.cell_nr))
        for t in range(map.shape[0]):
            map[t][(map_id_state[t, :, 1] - 1).astype(int)] = map_id_state[t, :, -1]

        cell_state = np.zeros(cell_id.shape)
        for t in range(map.shape[0]):
            cell_state[t] = map[t][cell_id[t].astype(int) - 1]
        cell_state = np.where(cell_id == 0, 0, cell_state)

        cell_mask = np.zeros(cell_id.shape)
        cell_mask = np.where(cell_id != 0, 1, 0)

        return cell_state, cell_mask

    def read_2d_data(self, path: str):
        v_ext = self.__read_V_2d__(path)
        cell_state, cell_mask = self.__read_I_2d__(path)
        data = np.concatenate(
            [
                np.expand_dims(v_ext, axis=3),
                np.expand_dims(cell_state, axis=3),
                np.expand_dims(cell_mask, axis=3),
            ],
            axis=3,
        )
        return data

    @timeit
    def read_offline_data_2d(self, path: str, workdir):
        with open(os.path.join(workdir, "log_read_offline.txt"), "w") as logfile:
            with redirect_stdout(logfile), redirect_stderr(logfile):
                path_list = glob.glob(path)
                nr_of_params = self.config.param_nr

                n_sim = len(path_list)
                dfs = np.empty(
                    (
                        n_sim,
                        self.config.timesteps
                        - 1
                        - self.config.cut_off_start
                        - self.config.cut_off_end,
                        self.config.grid_size,
                        self.config.grid_size,
                        self.config.nr_observables,
                    ),
                    dtype=np.float32,
                )
                params = np.empty((n_sim, nr_of_params), dtype=np.float32)
                invalidIndices = []

                for path in tqdm(range(n_sim)):
                    pathname = path_list[path]
                    print(pathname)
                    path_split = pathname.split("/")[len(pathname.split("/")) - 1]
                    if "e" in path_split:
                        invalidIndices.append(path)
                        continue
                    if path_split.startswith("sweep") or path_split.startswith("DV"):
                        start_nr = 1
                    else:
                        start_nr = 0
                    params_split = path_split.split("_")[start_nr : nr_of_params + 1]
                    param_file = list(
                        map(lambda x: round(float(x.split("-")[1]), 3), params_split)
                    )
                    if self.config.rejectionBasedOnPrior(param_file) == True:
                        invalidIndices.append(path)
                        continue

                    sim = self.read_2d_data(pathname)
                    if self.config.rejectionBasedOnSimulation(sim):
                        invalidIndices.append(path)
                        continue

                    params[path] = param_file
                    dfs[path] = sim

                dfs = np.delete(dfs, invalidIndices, axis=0)
                params = np.delete(params, invalidIndices, axis=0)
                print("Read data in the form of: ", dfs.shape)

                return dfs, params
