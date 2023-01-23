import os
import numpy as np
from matplotlib import pyplot as plt


class ResultLogger:
    def __init__(
        self,
        workdir,
        trainer,
        losses,
        config,
        prior,
        diag,
        model,
        configurator,
        amortizer,
        testdata=None,
    ):
        self.workdir = workdir
        self.trainer = trainer
        self.losses = losses
        self.config = config
        self.prior = prior
        self.diag = diag
        self.model = model
        self.configurator = configurator
        self.amortizer = amortizer
        if testdata != None:
            self.testdata = testdata

        self.output_dir = os.path.abspath(os.path.join(workdir, config.plots))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.prior_means, self.prior_stds = prior.estimate_means_and_stds()

    def __runResimulations__(self):
        res = {}
        res["raw_sims"] = self.model(
            batch_size=self.config.resimulation_param["simulations"]
        )
        res["validation_sims"] = self.configurator(res["raw_sims"])
        res["post_samples"] = self.amortizer.sample(
            res["validation_sims"], self.config.resimulation_param["post_samples"]
        )
        res["post_samples_unnorm"] = (
            self.prior_means + res["post_samples"] * self.prior_stds
        )
        return res

    def __format_resimulation__(self):
        res = {}
        res["raw_sims"] = self.testdata
        res["validation_sims"] = self.configurator(res["raw_sims"])
        res["post_samples"] = self.amortizer.sample(
            res["validation_sims"], self.config.resimulation_param["post_samples"]
        )
        res["post_samples_unnorm"] = (
            self.prior_means + res["post_samples"] * self.prior_stds
        )
        return res

    def __generate_posterior_plots__(self, resimulations):
        prior_draws = resimulations["raw_sims"]["prior_draws"]
        post_mean = np.mean(resimulations["post_samples_unnorm"], axis=1)
        post_sd = np.std(resimulations["post_samples_unnorm"], axis=1)
        z_scores = (post_mean - prior_draws) / post_sd
        post_contraction = 1 - (post_sd / self.prior_stds) ** 2
        fig = plt.figure(1)
        cols = 1
        rows = len(self.config.prior_names)
        for p in range(1, rows + 1):
            ax = fig.add_subplot(rows, cols, p)
            ax.scatter(
                x=post_contraction[:, p - 1],
                y=z_scores[:, p - 1],
                c=prior_draws[:, p - 1],
            )
            ax.set_title(self.config.prior_names[p - 1])
            ax.set_xlim([0, 1])
            ax.set_ylim([-6, 6])
        return fig

    def create_plots(self):
        plot_dir = os.path.abspath(os.path.join(self.workdir, self.config.plots))
        if self.config.losses:
            loss = self.diag.plot_losses(
                self.losses["train_losses"], self.losses["val_losses"]
            )
            loss.savefig(os.path.join(plot_dir, "losses.png"))
            plt.close(loss)

        if self.config.latent2d:
            latent2d = self.trainer.diagnose_latent2d()
            latent2d.savefig(os.path.join(plot_dir, "latent2d.png"))
            plt.close(latent2d)

        if self.config.sbc_histograms:
            sbc_histograms = self.trainer.diagnose_sbc_histograms()
            sbc_histograms.savefig(os.path.join(plot_dir, "sbc_histograms.png"))
            plt.close(sbc_histograms)

        if self.config.run_resimulations:
            if self.config.test_online:
                res = self.__runResimulations__()
            else:
                res = self.__format_resimulation__()

        if self.config.plot_ppc:
            plot_ppc = self.__generate_posterior_plots__(res)
            plot_ppc.savefig(os.path.join(plot_dir, "plot_ppc.png"))
            plt.close(plot_ppc)

        if self.config.sbc_test_data:
            sbc_test = self.diag.plot_sbc_histograms(
                res["post_samples"],
                res["validation_sims"]["parameters"],
                self.config.prior_names,
                num_bins=10,
            )
            sbc_test.savefig(os.path.join(plot_dir, "sbc_test_data.png"))
            plt.close(sbc_test)

        if self.config.sbc_ecdf:
            sbc_ecdf = self.diag.plot_sbc_ecdf(
                res["post_samples"], res["validation_sims"]["parameters"]
            )
            sbc_ecdf.savefig(os.path.join(plot_dir, "sbc_ecdf.png"))
            plt.close(sbc_ecdf)

            sbc_ecdf_stacked = self.diag.plot_sbc_ecdf(
                res["post_samples"],
                res["validation_sims"]["parameters"],
                stacked=True,
                difference=True,
            )
            sbc_ecdf_stacked.savefig(os.path.join(plot_dir, "sbc_ecdf_stacked.png"))
            plt.close(sbc_ecdf_stacked)

        # TODO: posterior scores, correlation

        if self.config.recovery:
            recovery = self.diag.plot_recovery(
                res["post_samples"],
                res["validation_sims"]["parameters"],
                param_names=self.config.prior_names,
            )
            recovery.savefig(os.path.join(plot_dir, "recovery.png"))
            plt.close(recovery)

        if self.config.plot_individual_sims:
            for i in range(0, self.config.nr_individual_sims):
                posterior_2d = self.diag.plot_posterior_2d(
                    res["post_samples_unnorm"][i], self.prior
                )
                posterior_2d.savefig(os.path.join(plot_dir, "posterior_2d_%s.png" % i))
                plt.close(posterior_2d)

                if self.config.resimulation_plots:
                    resim = self.plot_resimulation(
                        res["raw_sims"]["sim_data"][i],
                        res["post_samples_unnorm"][
                            i, 0 : self.config.nr_of_resimulations
                        ],
                    )
                    resim.savefig(os.path.join(plot_dir, "resimulation_%s.png" % i))
                    plt.close(resim)

    def plot_resimulation(self, raw_sims, post_samples):  # TODO: generalisieren
        cut_timesteps = (
            self.config.timesteps
            - self.config.cut_off_end
            - self.config.cut_off_start
            - 1
        )
        nr_of_resimulations = self.config.nr_of_resimulations

        resim_I = np.empty((nr_of_resimulations, cut_timesteps), dtype=np.float32)
        resim_V = np.empty((nr_of_resimulations, cut_timesteps), dtype=np.float32)
        time_range = range(0, cut_timesteps)

        resim = self.trainer.generative_model.simulator.__call__(
            params=np.array(post_samples)
        )

        resim_I[:, :] = resim["sim_data"][:, :, 1]
        resim_V[:, :] = resim["sim_data"][:, :, 2]

        I_qt_50 = np.quantile(resim_I, q=[0.25, 0.75], axis=0)
        I_qt_90 = np.quantile(resim_I, q=[0.05, 0.95], axis=0)
        I_qt_95 = np.quantile(resim_I, q=[0.025, 0.975], axis=0)
        V_qt_50 = np.quantile(resim_V, q=[0.25, 0.75], axis=0)
        V_qt_90 = np.quantile(resim_V, q=[0.05, 0.95], axis=0)
        V_qt_95 = np.quantile(resim_V, q=[0.025, 0.975], axis=0)

        fig, ax = plt.subplots(2)
        ax[0].plot(
            time_range, np.median(resim_I, axis=0), label="Median u(t)", color="b"
        )
        ax[0].plot(
            time_range,
            raw_sims[:, 1],
            marker="o",
            label="Ground truth u(t)",
            color="k",
            linestyle="--",
            alpha=0.8,
        )
        ax[0].fill_between(
            time_range, I_qt_50[0], I_qt_50[1], color="b", alpha=0.3, label="50% CI"
        )
        ax[0].fill_between(
            time_range, I_qt_90[0], I_qt_90[1], color="b", alpha=0.2, label="90% CI"
        )
        ax[0].fill_between(
            time_range, I_qt_95[0], I_qt_95[1], color="b", alpha=0.1, label="95% CI"
        )
        ax[0].grid(True)
        ax[0].set_title("Re-simulation for infected cells")
        ax[0].set_xlabel("Time t [au]")
        ax[0].set_ylabel("Infected cells")
        ax[0].legend()

        ax[1].plot(
            time_range, np.median(resim_V, axis=0), label="Median v(t)", color="b"
        )
        ax[1].plot(
            time_range,
            raw_sims[:, 2],
            marker="o",
            label="Ground truth v(t)",
            color="k",
            linestyle="--",
            alpha=0.8,
        )
        ax[1].fill_between(
            time_range, V_qt_50[0], V_qt_50[1], color="b", alpha=0.3, label="50% CI"
        )
        ax[1].fill_between(
            time_range, V_qt_90[0], V_qt_90[1], color="b", alpha=0.2, label="90% CI"
        )
        ax[1].fill_between(
            time_range, V_qt_95[0], V_qt_95[1], color="b", alpha=0.1, label="95% CI"
        )
        ax[1].grid(True)
        ax[1].set_title("Re-simulation for virus particles")
        ax[1].set_xlabel("Time t [au]")
        ax[1].set_ylabel("Virus particles")
        ax[1].legend()
        return fig
