import tensorflow as tf


class ConvLSTM(tf.keras.Model):
    def __init__(self, n_summary):
        super(ConvLSTM, self).__init__()

        self.conv = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    n_summary,
                    kernel_size=3,
                    strides=1,
                    padding="causal",
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                ),
                tf.keras.layers.Conv1D(
                    n_summary * 2,
                    kernel_size=2,
                    strides=1,
                    padding="causal",
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                ),
                tf.keras.layers.Conv1D(
                    n_summary * 3,
                    kernel_size=1,
                    strides=1,
                    padding="causal",
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                ),
            ]
        )
        self.lstm = tf.keras.layers.LSTM(n_summary)
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_summary * 2, activation="relu"),
                tf.keras.layers.Dense(n_summary, activation="relu"),
            ]
        )

    def call(self, x, **args):
        """x is a 3D tensor of shape (batch_size, n_time_steps, n_time_series)"""
        out = self.conv(x)
        out = self.lstm(out)
        out = self.dense(out)

        return out


class ConvLSTM3D(tf.keras.Model):
    def __init__(self, n_summary):
        super(ConvLSTM3D, self).__init__()

        self.conv = tf.keras.Sequential(
            [
                tf.keras.layers.Conv3D(
                    n_summary,
                    input_shape=(30, 45, 45, 3),
                    data_format="channels_last",
                    kernel_size=3,
                    strides=1,
                    padding="valid",
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                ),
                tf.keras.layers.MaxPooling3D(
                    pool_size=(1, 2, 2), strides=None, padding="valid"
                )
                # ),
                # tf.keras.layers.Conv3D(
                #     n_summary * 2,
                #     input_shape=(28, 21, 21, 16),
                #     data_format="channels_last",
                #     kernel_size=3,
                #     strides=1,
                #     padding="valid",
                #     activation="relu",
                #     kernel_initializer="glorot_uniform",
                # ),
                # tf.keras.layers.MaxPooling3D(
                #     pool_size=(1, 2, 2), strides=None, padding="valid"
                # ),
                # tf.keras.layers.Conv3D(
                #     n_summary * 3,
                #     input_shape=(26, 10, 10, 32),
                #     data_format="channels_last",
                #     kernel_size=3,
                #     strides=1,
                #     padding="valid",
                #     activation="relu",
                #     kernel_initializer="glorot_uniform",
                # ),
                # tf.keras.layers.MaxPooling3D(
                #     pool_size=(1, 2, 2), strides=None, padding="valid"
                # ),
            ]
        )
        self.reshape = tf.keras.layers.Reshape((28, 21 * 21 * 16))
        self.lstm = tf.keras.layers.LSTM(n_summary)
        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_summary, activation="relu"),
            ]
        )

    def call(self, x, **args):
        """x is a 3D tensor of shape (batch_size, n_time_steps, n_time_series)"""
        out = self.conv(x)
        out = self.reshape(out)
        out = self.lstm(out)
        out = self.dense(out)

        return out
