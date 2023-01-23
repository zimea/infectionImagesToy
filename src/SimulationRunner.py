import glob
import os, sys
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import time
from pandas import read_csv
from Utility import timeit
from contextlib import redirect_stdout, redirect_stderr


class SimulationRunner:
    def __init__(self, config, workdir, dataReader):
        self.config = config
        self.workdir = workdir
        self.dataReader = dataReader

    @timeit
    def run_morpheus(self, params):
        with open(os.path.join(self.workdir, "log_morpheus.txt"), "w") as logfile:
            with redirect_stderr(logfile), redirect_stdout(logfile):
                bcf, pi = params
                cv = 0.33
                DV_str = "wm"
                DV = 999

                model_dir = "model"
                model_pattern = os.path.join(
                    self.config.data_path, model_dir, "*DV-%s*.xml" % DV_str
                )
                models = glob.glob(model_pattern)
                model = models[0]

                OUT = (
                    os.path.join(
                        self.config.data_path,
                        self.config.folder,
                        "DV-" + str(DV) + "_bcf-" + str(bcf) + "_" + "pi-" + str(pi),
                    )
                    + "_"
                    + "cv-"
                    + str(cv)
                )
                create_dir = Popen(
                    "mkdir " + OUT, shell=True, stdout=sys.stdout, stderr=sys.stderr
                )
                create_dir.wait()
                morpheus_command = (
                    "morpheus"
                    + " -f "
                    + model
                    + " -o "
                    + OUT
                    + " -b_cf="
                    + str(bcf)
                    + " -c_V="
                    + str(cv)
                    + " -p_V="
                    + str(pi)
                )
                print(morpheus_command)

                run_sim = Popen(
                    morpheus_command, shell=True, stdout=sys.stdout, stderr=sys.stderr
                )
                run_sim.wait()

                final_plot = os.path.join(
                    OUT, "plot_" + str(self.config.timesteps).zfill(5) + ".png"
                )
                while not os.path.exists(final_plot):
                    time.sleep(1)

                population_file = os.path.join(OUT, "logger_2.csv")
                df = read_csv(population_file, sep="\t")
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

                v_path = os.path.join(OUT, "logger_6_Ve.csv")
                v = self.dataReader.calculate_V(v_path)
                sim = np.append(df_cells, v, axis=1)
                # I_volume = self.dataReader.calculate_volume(OUT)[
                #     self.config.cut_off_start
                #     + 1 : self.config.timesteps
                #     - self.config.cut_off_end
                # ]
                # sim = np.append(sim, I_volume, axis=1)

                return sim

    @timeit
    def run_morpheus_2d(self, params):
        with open(os.path.join(self.workdir, "log_morpheus.txt"), "w") as logfile:
            with redirect_stderr(logfile), redirect_stdout(logfile):
                bcf, pi = params
                cv = 0.33
                DV_str = "wm"
                DV = 999

                model_dir = "model"
                model_pattern = os.path.join(
                    self.config.data_path, model_dir, "*DV-%s*.xml" % DV_str
                )
                models = glob.glob(model_pattern)
                model = models[0]

                OUT = (
                    os.path.join(
                        self.config.data_path,
                        self.config.folder,
                        "DV-" + str(DV) + "_bcf-" + str(bcf) + "_" + "pi-" + str(pi),
                    )
                    + "_"
                    + "cv-"
                    + str(cv)
                )
                create_dir = Popen(
                    "mkdir " + OUT, shell=True, stdout=sys.stdout, stderr=sys.stderr
                )
                create_dir.wait()
                morpheus_command = (
                    "morpheus"
                    + " -f "
                    + model
                    + " -o "
                    + OUT
                    + " -b_cf="
                    + str(bcf)
                    + " -c_V="
                    + str(cv)
                    + " -p_V="
                    + str(pi)
                )
                print(morpheus_command)

                run_sim = Popen(
                    morpheus_command, shell=True, stdout=sys.stdout, stderr=sys.stderr
                )
                run_sim.wait()

                final_plot = os.path.join(
                    OUT, "plot_" + str(self.config.timesteps).zfill(5) + ".png"
                )
                while not os.path.exists(final_plot):
                    time.sleep(1)

                sim = self.dataReader.read_2d_data(OUT)

                return sim
