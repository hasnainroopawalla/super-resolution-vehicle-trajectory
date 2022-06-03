from tensorflow.keras.models import Model
import tensorflow as tf

from rits import RITS


class BRITS(Model):
    def __init__(self, internal_dim, hid_dim, sequence_length=None, name='BRITS'):
        super(BRITS, self).__init__(name=name)
        self.hid_dim = hid_dim
        self.internal_dim = internal_dim
        self.sequence_length = sequence_length
        return

    def build(self, input_shape):
        self.sequence_length = input_shape[1]
        self.rits_f = RITS(self.internal_dim, self.hid_dim, self.sequence_length, go_backwards=False, name="RITS_F")
        self.rits_b = RITS(self.internal_dim, self.hid_dim, self.sequence_length, go_backwards=False, name="RITS_B")
        return

    def reverse(self, imputations):
        return imputations[:, ::-1, :]

    def call(self, values, masks, deltas):
        imputations_f, custom_loss_f = self.rits_f(values, masks, deltas)
        imputations_b, custom_loss_b = self.rits_b(values, masks, deltas)

        imputations_b = self.reverse(imputations_b)

        imputations = (imputations_f + imputations_b) / 2.0

        discrepancy_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(imputations_f, imputations_b), axis=1) + tf.reduce_mean(tf.keras.losses.mean_squared_error(imputations_f, imputations_b), axis=1)
        return imputations, custom_loss_f + custom_loss_b + discrepancy_loss
