import argparse
import sys, os
import importlib.util
import time

parser = argparse.ArgumentParser(
    prog="ExperimentRunner",
    description="Starts individual experiments and logs results",
)
parser.add_argument(
    "-w",
    "--workdir",
    type=str,
    help="path to the working directory for input and output",
    default="/home/l/projects/Morpheus/Tutorial/Experiments/experiments/offline_1000_double_summary/",
)
parser.add_argument(
    "-c", "--configfile", type=str, help="path to the config file", default="config.py"
)
args = parser.parse_args()
workdir = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), args.workdir)))
configfile = os.path.abspath(os.path.abspath(os.path.join(workdir, args.configfile)))

spec = importlib.util.spec_from_file_location("config", configfile)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

sys.path.append(os.path.abspath(os.path.join(config.SourcePath)))
sys.path.append(os.path.abspath(os.path.join(config.BayesFlowPath)))
from MorpheusReader import MorpheusReader
from SimulationRunner import SimulationRunner
from ResultLogger import ResultLogger
from Utility import inject_class_method
from functools import partial
import tensorflow as tf
from contextlib import redirect_stdout, redirect_stderr
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.trainers import Trainer
import bayesflow.diagnostics as diag

# Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)


if __name__ == "__main__":
    with open(os.path.join(workdir, "log_training.txt"), "w") as logfile:
        with redirect_stdout(logfile), redirect_stderr(logfile):
            logfile.write("Initialize prior\n")
            prior = Prior(prior_fun=config.prior_func, param_names=config.prior_names)
            prior_means, prior_stds = prior.estimate_means_and_stds()
            dataReader = MorpheusReader(
                config, prior_means=prior_means, prior_stds=prior_stds
            )
            logfile.write("Initialize generative model\n")
            simulationRunner = SimulationRunner(config, workdir, dataReader)

            if config.image_data:
                sim_func = simulationRunner.run_morpheus_2d
            else:
                sim_func = simulationRunner.run_morpheus
            simulator = Simulator(simulator_fun=partial(sim_func))
            model = GenerativeModel(prior, simulator, name=config.model_name)

            logfile.write("Initialize amortizer\n")
            summary_net = config.summary_network
            inference_net = InvertibleNetwork(
                num_params=config.param_nr, num_coupling_layers=config.inn_layer
            )
            amortizer = AmortizedPosterior(
                inference_net, summary_net, name=config.amortizer_name
            )
            logfile.write("Initialize trainer\n")
            trainer = Trainer(
                amortizer=amortizer,
                generative_model=model,
                configurator=dataReader.prepare_input,
                checkpoint_path=os.path.join(workdir, config.checkpoints),
                optional_stopping=config.optional_stopping,
            )
            logfile.write("Initialization finished\n")

            match config.training_mode:
                case "offline":
                    logfile.write("Start reading data\n")
                    train, test, validation = dataReader.read_offline_split(
                        os.path.join(config.data_path, config.folder + "/*"), workdir
                    )
                    logfile.write("Finished reading data\n")
                    logfile.write("Start training\n")
                    start_time = time.time()
                    h = trainer.train_offline(
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        simulations_dict=train,
                        validation_sims=validation,
                    )
                    end_time = time.time()
                    elapsed_time = time.time() - start_time
                    logfile.write("Finished training\n")
                    logfile.write(str(elapsed_time))
                case "online":
                    h = trainer.train_online(
                        epochs=config.epochs,
                        iterations_per_epoch=config.iter_per_epoch,
                        batch_size=config.batch_size,
                        validation_sims=config.validation_nr,
                    )
                case _:
                    logfile.write("Unbekannter Trainingsmodus\n")
                    sys.exit()

            print("Plot results")

            results = ResultLogger(
                workdir=workdir,
                trainer=trainer,
                losses=h,
                config=config,
                prior=prior,
                diag=diag,
                model=model,
                configurator=dataReader.prepare_input,
                amortizer=amortizer,
                testdata=test,
            )
            results.create_plots()
